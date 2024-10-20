import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from matplotlib.lines import Line2D


# Define the simplified FNN model with Dropout and Batch Normalization
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer with 20% drop rate

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)  # Apply dropout
        x = self.output_layer(x)
        return x


# Define Custom Loss Function
class CustomLoss(nn.Module):
    def __init__(self, weights):
        super(CustomLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        assert outputs.shape[1] == 3, "Outputs and weights must be for 3 elements"
        component_losses = []
        total_loss = 0
        for i in range(3):
            loss = (outputs[:, i] - targets[:, i]) ** 2  # MSE for each component
            weighted_loss = self.weights[i] * loss.mean()  # Weighted loss
            component_losses.append(weighted_loss.item())  # Store individual component loss
            total_loss += weighted_loss
        return total_loss, component_losses


# Load and prepare data
def load_and_prepare_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df['next_episode'] = df['episode'].shift(-1, fill_value=df['episode'].iloc[-1])
    df = df[df['episode'] == df['next_episode']]
    df.drop(columns=['next_episode'], inplace=True)

    features = ['prev_phase_value_x', 'prev_phase_value_y',
                'prev_amplitude_value_x', 'prev_amplitude_value_y',
                'normalized_prev_angle', 'prev_current_center_x', 'prev_current_center_y',
                'normalized_causing_action']


    targets = ['next_current_center_x', 'next_current_center_y', 'normalized_next_angle']

    df[features[:-3]] = df[features[:-3]].round(3)
    df['normalized_next_angle'] = df['normalized_next_angle'].round(3)
    df['next_current_center_x'] = df['next_current_center_x'].round(3)
    df['next_current_center_y'] = df['next_current_center_y'].round(3)
    df['normalized_prev_angle'] = df['normalized_prev_angle'].round(3)

    X = df[features].values[:-1]
    y = df[targets].values[1:]

    return X, y

# Train the model with early stopping and save the ensemble models
def train_ensemble_with_bagging(X_train, y_train, X_val, y_val, num_models=5, num_epochs=2000, hidden_dim=32, lr=0.001, weights=[1, 1, 1], patience=100):
    models = []
    for i in range(num_models):
        # Create a bootstrap sample
        bootstrap_indices = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=True)
        X_train_bootstrap = X_train[bootstrap_indices]
        y_train_bootstrap = y_train[bootstrap_indices]

        model = FNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim, output_dim=y_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = CustomLoss(weights)

        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in tqdm(range(num_epochs), desc=f"Training Model {i + 1}/{num_models}", unit="epoch"):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_bootstrap)
            loss, _ = loss_fn(outputs, y_train_bootstrap)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss, _ = loss_fn(val_outputs, y_val)

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'best_model_ensemble_{i}.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    tqdm.write(f'Early stopping triggered for model {i + 1} after {epoch + 1} epochs.')
                    break

        model.load_state_dict(torch.load(f'best_model_ensemble_{i}.pth'))
        models.append(model)

    return models



# Function to perform ensemble predictions
def ensemble_predict(models, X):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds = model(X).cpu().numpy()
        predictions.append(preds)

    predictions = np.array(predictions)
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)
    return pred_mean, pred_std


# Evaluate the ensemble model
def evaluate_ensemble(models, X_test_tensor, y_test_tensor, loss_fn):
    y_pred_mean, y_pred_std = ensemble_predict(models, X_test_tensor)

    # Convert predictions to tensors for loss calculation
    y_pred_mean_tensor = torch.FloatTensor(y_pred_mean).to(device)
    test_loss, test_component_losses = loss_fn(y_pred_mean_tensor, y_test_tensor)

    print(f'Test Loss: {test_loss}')
    print(f'Test Component Losses: {test_component_losses}')
    print(f'Prediction Uncertainty (Std): {y_pred_std.mean(axis=0)}')

    return test_loss, test_component_losses, y_pred_mean, y_pred_std

import optuna
def objective(trial, csv_file_path):
    # Define the hyperparameters to search
    hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    num_models = 5
    num_epochs = 1500
    weights = [trial.suggest_float(f'weight_{i}', 0.5, 2) for i in range(3)]

    X, y = load_and_prepare_data(csv_file_path)
    test_size = 0.2
    num_test_samples = int(len(X) * test_size)
    X_train = X[:-num_test_samples]
    y_train = y[:-num_test_samples]
    X_test = X[-num_test_samples:]
    y_test = y[-num_test_samples:]

    scaler_x = MinMaxScaler().fit(X_train)
    scaler_y = MinMaxScaler().fit(y_train)
    X_train_scaled = scaler_x.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

    # Train the ensemble of models
    models = train_ensemble_with_bagging(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
                            num_models=num_models, num_epochs=num_epochs, hidden_dim=hidden_dim, lr=lr, weights=weights)

    # Evaluate on test set with ensemble
    test_loss, _, y_pred_mean, y_pred_std = evaluate_ensemble(models, X_test_tensor, y_test_tensor, CustomLoss(weights))

    # Calculate the metric to optimize (std of ensemble predictions)
    metric = y_pred_std.mean()

    return metric

def run_optuna(csv_file_path):
    # Run Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, csv_file_path), n_trials=100, show_progress_bar=True)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    #save the best trials to a text file
    with open("best_trials.txt", "w") as file:
        file.write(f"Best trial value: {trial.value}\n")
        file.write("Best trial parameters:\n")
        for key, value in trial.params.items():
            file.write(f"{key}: {value}\n")

    return study


# Additional function to calculate confidence intervals
def confidence_intervals(mean, std_dev, z=1.96):
    ci_lower = mean - z * std_dev
    ci_upper = mean + z * std_dev
    return ci_lower, ci_upper

# Load necessary libraries for metrics
def compute_metrics(y_true, y_pred, y_pred_std):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    exp_var = explained_variance_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    std_mean = np.mean(y_pred_std)  # Average standard deviation over the predictions
    return rmse, mae, exp_var, r2, std_mean



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_file_path = 'output_data_5_26.csv'

    X, y = load_and_prepare_data(csv_file_path)

    # Load test data and preprocess
    test_size = 0.2
    num_test_samples = int(len(X) * test_size)
    X_train = X[:-num_test_samples]
    y_train = y[:-num_test_samples]
    X_test = X[-num_test_samples:]
    y_test = y[-num_test_samples:]

    scaler_x = MinMaxScaler().fit(X_train)
    scaler_y = MinMaxScaler().fit(y_train)
    X_train_scaled = scaler_x.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    # Save final test scalers
    dump(scaler_x, 'scaler_x_test.joblib')
    dump(scaler_y, 'scaler_y_test.joblib')

    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

    # Train the ensemble of models with bagging
    models = train_ensemble_with_bagging(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_models=5, num_epochs=2000, hidden_dim=121, lr=0.009936890104195635, weights=[1.3110366957384838, 1.2525048455822758, 1.7789145100613415], patience=100)

    # Evaluate on test set with ensemble
    test_loss, test_component_losses, y_pred_mean, y_pred_std = evaluate_ensemble(
        models, X_test_tensor, y_test_tensor,
        CustomLoss(weights=[1.3110366957384838, 1.2525048455822758, 1.7789145100613415])
    )

    true_values = y_test_tensor.cpu().numpy()
    ci_lower, ci_upper = confidence_intervals(y_pred_mean, y_pred_std)

    targets = ['Next Robot Center X', 'Next Robot Center Y', 'Next Robot Angle']

    import matplotlib as mpl

    # Set global font sizes using rcParams
    mpl.rcParams.update({'font.size': 22, 'axes.labelsize': 26, 'axes.titlesize': 26,
                         'xtick.labelsize': 30, 'ytick.labelsize': 30, 'legend.fontsize': 20})

    # Adjust these dimensions to find the best fit for your output medium
    plt.figure(figsize=(34, 18))  # Moderate figure size for clarity

    targets = ['Next Robot Center X', 'Next Robot Center Y', 'Next Robot Angle']

    for i, target in enumerate(targets):
        rmse, mae, exp_var, r2, std_mean = compute_metrics(true_values[:, i], y_pred_mean[:, i], y_pred_std[:, i])
        residuals = true_values[:, i] - y_pred_mean[:, i]

        ax1 = plt.subplot(2, 3, i + 1)
        plt.errorbar(true_values[:, i], y_pred_mean[:, i],
                     yerr=[y_pred_mean[:, i] - ci_lower[:, i], ci_upper[:, i] - y_pred_mean[:, i]], fmt='o',
                     color='blue', ecolor='darkblue', elinewidth=1.5, alpha=0.7, capsize=6)
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

    # Adding legends
    prediction_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12,
                               label='Predictions with 95% CI')
    ideal_line = Line2D([0], [0], linestyle='--', color='red', label='Ideal Prediction')
    residual_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Residuals')

    plt.tight_layout(pad=3.0)  # Adjust overall layout padding
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'Next_State_Prediction_Model_{current_time}.jpg', dpi=600)  # Adjust DPI for different quality needs
    plt.show()

    # Run Optuna optimization
    # run_optuna(csv_file_path)
    # #best trial:
    #
    # print("Optuna optimization completed.")


