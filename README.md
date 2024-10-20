# Magnetic Soft Robot Closed Loop Control Framework: Model Inference via Reinforcement Learning and Multi-Layer Perceptron Ensemble

This repository contains the code and data used in the paper "Magnetic Soft Robot Closed Loop Control Framework: Model Inference via Reinforcement Learning and Multi-Layer Perceptron Ensemble" submitted to the IEEE Transactions on Mechantronics.

## Authors

- Zakir Ullah
- Dong Wang
- Jinqiang Wang
- Zixiao Zhu
- Peter B. Shull

## Repository Structure

### Folders

#### `train_unet_model`
- **Description**: Contains the dataset and segmentation model code to generate a model for robot image segmentation.
- **Usage**: Use this folder to train and evaluate your UNet-based image segmentation model.

#### `MLP_ensemble_model`
- **Description**: This folder includes scripts to create MLP ensemble models and generate scalar files based on the dataset `output_data_5_26`.
- **Usage**: Use the scripts in this folder to train MLP ensemble models and prepare them for predicting robot actions.

#### `camera`
- **Description**: Contains scripts and utilities related to camera interface and image acquisition for segmentation tasks.
- **Usage**: Use this folder to set up and manage the camera interface for capturing images used in segmentation.

### Files

#### `FY8300.py`
- **Description**: Library file for the FY8300 signal generator.
- **Usage**: Utilize this library to interface with and control the FY8300 signal generator.

#### `interface.py`
- **Description**: Applies model predictive control (MPC) to the robot.
- **Usage**: Run this script to implement MPC on your robot for various control tasks.

#### `New_lowlevel.py`
- **Description**: Represents the environment for reinforcement learning (RL) training.
- **Usage**: This script defines the RL training environment. Use it in conjunction with `New_train_rl.py` for training.

#### `New_train_rl.py`
- **Description**: The training script for the RL environment.
- **Usage**: Run this script to train your RL model using the environment defined in `New_lowlevel.py`.

#### `Robot_camera.py`
- **Description**: Manages the camera interface for the robot.
- **Usage**: Use this script to handle image capturing and processing tasks with the robot's camera.

#### `predict_image.py`
- **Description**: Responsible for image segmentation tasks.
- **Usage**: This script performs image segmentation using the trained model from `train_unet_model`.

#### `predic_action.py`
- **Description**: Implements the A* algorithm infused with the MLP model to predict the actions required for the robot to reach a target.
- **Usage**: Run this script to calculate the optimal path and actions for the robot using the A* algorithm and MLP predictions.

### Model Files

#### `best_model_ensemble_X.pth` (where X is 0, 1, 2, 3, 4)
- **Description**: Saved models from the MLP ensemble training.
- **Usage**: Use these model files to load pre-trained MLP ensemble models for inference tasks.

#### `scaler_x_test.joblib` & `scaler_y_test.joblib`
- **Description**: Scalar files for input normalization used in conjunction with the MLP models.
- **Usage**: Load these scalars to normalize input data before feeding it into the MLP models.


