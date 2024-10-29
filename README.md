
# Model-Free Closed-Loop Control of Magnetic Soft Robots via Reinforcement Learning and Ensemble Neural Networks

This repository contains the code and data for the paper **"Model-Free Closed-Loop Control of Magnetic Soft Robots via Reinforcement Learning and Ensemble Neural Networks."**

## Authors

- Zakir Ullah
- Dong Wang
- Jinqiang Wang
- Zixiao Zhu
- Peter B. Shull

## Repository Structure

### Main Files

- **Generate_experimental_data.py**: Script to generate experimental data for the reinforcement learning environment, using PPO to train a policy in a custom magnetic robot environment.
- **Implementation_PAC.py**: Implementation of Proximal Action Control (PAC) for low-level control of magnetic soft robots, integrating hardware control for signal generation and real-time image processing.
- **Markov_Env.py**: Custom reinforcement learning environment extending OpenAI Gym to simulate robot motion under magnetic control, with customizable action and observation spaces.

### Project Directories

- **Mlp_ensemble_model/**: Contains scripts and utilities to train an ensemble model for robot state estimation, with file details as follows:
  - **main.py**: Initializes and trains an ensemble of neural networks.
  - **model.py**: Defines the Feedforward Neural Network (FNN) architecture.
  - **util.py**: Provides utility functions for loading data, model evaluation, and metric calculations.
  - **README_ensemble.md**: Documentation for the Mlp_ensemble_model subfolder.
  - **data/**: Contains sample data, including `output_data_5_26.csv` used for training and testing.
  - **results/**: Stores model checkpoints and scaler files for prediction standardization.

- **train_unet_model/**: Includes scripts for training a U-Net model to segment robot parts in images:
  - **dataset.py**: Handles data loading and processing.
  - **model.py**: Defines the U-Net architecture.
  - **train.py**: Manages the training loop, including data augmentation.
  - **util.py**: Provides utilities for saving results and plotting training metrics.
  - **results/**: Stores model checkpoints and generated visualizations.
  - **multilabel_dataset1_2/**: Contains example images and annotations in JSON format.

- **camera/**: Includes **camera.py**, a module for camera-based image capture and processing.

- **utils/**: Contains additional utility scripts:
  - **FY8300.py**: Interface for controlling an FY8300 signal generator.
  - **predict_image.py**: Script for processing images and making predictions.
  - **predic_action.py**: Implements action prediction logic.
  - **Robot_camera.py**: Utility for capturing images from a robotic camera.

### Configuration and Metadata

- **.github/ISSUE_TEMPLATE/**: Templates for creating bug reports, feature requests, and custom issues.
- **.idea/**: Project settings for integrated development environments (IDEs) like PyCharm.

### Other Files

- **requirements.txt**: List of dependencies to run the project.
- **scaler_x_test.joblib** and **scaler_y_test.joblib**: Scalers used for normalizing input and output data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zakir300408/Model_free_control_for_MSRs.git
   cd Model_free_control_for_MSRs
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure hardware components (camera, signal generator) are connected if using real-time control.

## Usage

1. **Experimental Data Generation**: Run `Generate_experimental_data.py` to collect data for the robot environment.
2. **Ensemble Model Training**: Navigate to `Mlp_ensemble_model/` and execute `main.py` to train the ensemble model.
3. **U-Net Model Training**: Navigate to `train_unet_model/` and execute `train.py` to start segmentation model training.
4. **Robot Control**: Use `Implementation_PAC.py` to execute the Proximal Action Control for hardware-integrated robot movement.

For detailed usage and customization instructions, refer to the README in each subfolder.

## Contact

For questions, contact Zakir Ullah at zakirullahmech@outlook.com.
