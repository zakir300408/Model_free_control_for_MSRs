import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

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


def compute_metrics(y_true, y_pred, y_pred_std):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    exp_var = explained_variance_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    std_mean = np.mean(y_pred_std)
    return rmse, mae, exp_var, r2, std_mean


def confidence_intervals(mean, std_dev, z=1.96):
    ci_lower = mean - z * std_dev
    ci_upper = mean + z * std_dev
    return ci_lower, ci_upper

def evaluate_ensemble(models, X_test_tensor, y_test_tensor, loss_fn):
    y_pred_mean, y_pred_std = ensemble_predict(models, X_test_tensor)

    # Convert predictions to tensors for loss calculation
    y_pred_mean_tensor = torch.FloatTensor(y_pred_mean).to(X_test_tensor.device)
    test_loss, test_component_losses = loss_fn(y_pred_mean_tensor, y_test_tensor)

    print(f'Test Loss: {test_loss}')
    print(f'Test Component Losses: {test_component_losses}')
    print(f'Prediction Uncertainty (Std): {y_pred_std.mean(axis=0)}')

    return test_loss, test_component_losses, y_pred_mean, y_pred_std
