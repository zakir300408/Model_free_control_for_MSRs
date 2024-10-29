import torch
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.lines import Line2D
import matplotlib as mpl
from model import FNN, CustomLoss, train_ensemble_with_bagging
from sklearn.preprocessing import MinMaxScaler

from util import load_and_prepare_data, compute_metrics, confidence_intervals, evaluate_ensemble
from joblib import dump


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_file_path = 'data/output_data_5_26.csv'
    X, y = load_and_prepare_data(csv_file_path)

    test_size = 0.2
    num_test_samples = int(len(X) * test_size)
    X_train, y_train = X[:-num_test_samples], y[:-num_test_samples]
    X_test, y_test = X[-num_test_samples:], y[-num_test_samples:]

    scaler_x = MinMaxScaler().fit(X_train)
    scaler_y = MinMaxScaler().fit(y_train)
    X_train_scaled = scaler_x.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    dump(scaler_x, 'results/scaler_x_test.joblib')
    dump(scaler_y, 'results/scaler_y_test.joblib')

    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

    models = train_ensemble_with_bagging(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_models=5, num_epochs=2000, hidden_dim=121, lr=0.009936890104195635, weights=[1.3110366957384838, 1.2525048455822758, 1.7789145100613415], patience=100)

    test_loss, test_component_losses, y_pred_mean, y_pred_std = evaluate_ensemble(
        models, X_test_tensor, y_test_tensor,
        CustomLoss(weights=[1.3110366957384838, 1.2525048455822758, 1.7789145100613415])
    )

    true_values = y_test_tensor.cpu().numpy()
    ci_lower, ci_upper = confidence_intervals(y_pred_mean, y_pred_std)

    targets = ['Next Robot Center X', 'Next Robot Center Y', 'Next Robot Angle']
    mpl.rcParams.update({'font.size': 22, 'axes.labelsize': 26, 'axes.titlesize': 26, 'xtick.labelsize': 30, 'ytick.labelsize': 30, 'legend.fontsize': 20})
    plt.figure(figsize=(34, 18))

    for i, target in enumerate(targets):
        rmse, mae, exp_var, r2, std_mean = compute_metrics(true_values[:, i], y_pred_mean[:, i], y_pred_std[:, i])
        residuals = true_values[:, i] - y_pred_mean[:, i]
        ax1 = plt.subplot(2, 3, i + 1)
        plt.errorbar(true_values[:, i], y_pred_mean[:, i], yerr=[y_pred_mean[:, i] - ci_lower[:, i], ci_upper[:, i] - y_pred_mean[:, i]], fmt='o', color='blue', ecolor='darkblue', elinewidth=1.5, alpha=0.7, capsize=6)
        plt.plot([0, 1], [0, 1], 'r--', zorder=5, linewidth=3)
        plt.xlabel('True Values', labelpad=10, fontsize=24)
        plt.ylabel('Predicted Values', labelpad=10, fontsize=24)
        plt.title(f'{target}\nRMSE: {rmse:.2f}, R²: {r2:.2f}, Exp Var: {exp_var:.2f}, Std Dev: {std_mean:.5f}', pad=30)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(0, 1.05)

        ax2 = plt.subplot(2, 3, i + 4)
        plt.scatter(true_values[:, i], residuals, color='Crimson', alpha=0.7)
        plt.axhline(0, color='black', linewidth=3)
        plt.xlabel('True Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Analysis for {target}')
        ax2.set_xlim(-0.05, 1)
        ax2.set_ylim(-1, 1)

    prediction_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12, label='Predictions with 95% CI')
    ideal_line = Line2D([0], [0], linestyle='--', color='red', label='Ideal Prediction')
    residual_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Residuals')
    plt.tight_layout(pad=3.0)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/ensemble_results_{current_time}.png")
    plt.show()
