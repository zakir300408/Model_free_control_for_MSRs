
# Ensemble Prediction for Robot State Estimation

Author: Zakir Ullah  
Email: zakirullahmech@outlook.com  

## Project Overview

This project trains an ensemble of Feedforward Neural Networks (FNNs) to estimate the next state (position and angle) of a robot given historical data. The model utilizes bagging to improve robustness and custom loss weights to handle multiple target variables.

## Files

- `main.py`: The main script to load data, prepare it, train the ensemble, and evaluate the model.
- `model.py`: Defines the FNN model and custom loss function, and implements ensemble training with bagging.
- `util.py`: Provides utility functions for data preparation, evaluation, and metrics calculation.

## Usage

1. **Data Preparation**: Place your data in `data/output_data_5_26.csv`. Ensure it contains the necessary columns for both features and targets.
2. **Training**: Run `main.py` to train the ensemble model. Results and visualizations will be saved in the `results` folder.
3. **Evaluation**: The model provides RMSE, R², explained variance, and other metrics, with predictions and confidence intervals plotted.

## File Modification Hints

- **`main.py`**:
  - Modify `csv_file_path` to set the path to your CSV data.
  - Adjust `test_size` if you want a different train-test split.
  - Modify `num_models`, `num_epochs`, `hidden_dim`, and `lr` when calling `train_ensemble_with_bagging` to experiment with ensemble size, training epochs, model complexity, and learning rate.
  - In `evaluate_ensemble`, modify `weights` in `CustomLoss` if the error scaling needs adjustment for each target.

- **`model.py`**:
  - Modify `input_dim` and `output_dim` in `FNN` if changing the number of features or targets.
  - Adjust `hidden_dim` in `train_ensemble_with_bagging` to control model capacity.
  - Update `weights` in `CustomLoss` to tune the penalty for each target in loss calculation.
  - If adding layers or dropout is necessary, adjust `FNN` as required.

- **`util.py`**:
  - Update `features` and `targets` lists in `load_and_prepare_data` to match your dataset.
  - Modify `ensemble_predict` and `evaluate_ensemble` if additional uncertainty metrics are required.
  - The `compute_metrics` function calculates RMSE, MAE, explained variance, and R²; update as needed for additional metrics.

## Contact

For questions, contact Zakir Ullah at zakirullahmech@outlook.com.
