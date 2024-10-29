import torch
from joblib import load
import numpy as np
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from joblib import load
import matplotlib.pyplot as plt
import time
import pickle


class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer with 20% drop rate

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
def denormalize_action(causing_action):
    causing_action = min(round(causing_action * 7), 7)
    if causing_action < 4:  # phase adjustments
        channel_index = causing_action // 2
        phase_value = [0, 180][causing_action % 2]
        param_type = 'phase'
    else:  # amplitude adjustments
        channel_index = (causing_action - 4) // 2
        amplitude_value = [0, 7][(causing_action - 4) % 2]
        param_type = 'amplitude'
    channel = ['X', 'Y'][channel_index]
    value = phase_value if param_type == 'phase' else amplitude_value

    # normalize the value
    value = value / 180 if param_type == 'phase' else value / 7
    return param_type, channel, value

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



# Load ensemble models
def load_ensemble_models(num_models=5, model_path_template='best_model_ensemble_{}.pth'):
    models = []
    for i in range(num_models):
        model = FNN(8, 121, 3).to(device)
        model.load_state_dict(torch.load(model_path_template.format(i), map_location=device))
        model.eval()
        models.append(model)
    return models
#the models are located here Mlp_ensemble_model/results
models = load_ensemble_models(model_path_template='Mlp_ensemble_model/results/best_model_ensemble_{}.pth')

scaler_x = load('Mlp_ensemble_model/results/scaler_x_test.joblib')
scaler_y = load('Mlp_ensemble_model/results/scaler_y_test.joblib')

# Define the causing action values
causing_action_values = np.linspace(0, 1, 8)

def transform_features(features, scaler, device):
    """ Scale features and convert them to a tensor. """
    features_scaled = scaler.transform(features)
    return torch.FloatTensor(features_scaled).to(device)

def predict_with_model(features_tensor, models, scaler_y):
    """ Evaluate the model and convert predictions back to original scale. """
    predictions = []
    for model in models:
        with torch.no_grad():
            predictions.append(model(features_tensor))
    predictions = torch.stack(predictions)
    pred_mean = torch.mean(predictions, dim=0)
    pred_std = torch.std(predictions, dim=0)
    # Convert predictions back to original scale
    pred_mean_np = pred_mean.cpu().numpy()
    pred_std_np = pred_std.cpu().numpy()
    return scaler_y.inverse_transform(pred_mean_np), scaler_y.inverse_transform(pred_std_np)

def batch_predict(features_array, models, device, scaler_x, scaler_y):
    """ Batch prediction for all features. """
    features_tensor = transform_features(features_array, scaler_x, device)
    return predict_with_model(features_tensor, models, scaler_y)

def predict_next_state_for_all_actions(initial_features, causing_action_values):
    start_time = time.time()
    all_predictions = {}

    num_actions = len(causing_action_values)
    all_features = np.tile(initial_features, (num_actions, 1))
    all_features[:, 7] = causing_action_values  # Direct assignment to the entire column

    # Perform batch prediction
    predictions_array, _ = batch_predict(all_features, models, device, scaler_x, scaler_y)  # Unpack mean predictions and ignore std

    # Calculate results
    for idx, causing_action in enumerate(causing_action_values):
        prediction = predictions_array[idx]
        # Round the prediction to 3 decimal places
        prediction = np.round(prediction, 3)
        x_position = prediction[0]  # Extract x position
        y_position = prediction[1]  # Extract y position
        angle = prediction[2]       # Extract angle
        distance = calculate_distance(np.round(all_features[idx][5], 2), np.round(all_features[idx][6], 2), x_position, y_position)
        all_predictions[causing_action] = {
            'causing_action': causing_action,
            'position': (x_position, y_position),
            'angle': angle,
            'distance_change': distance
        }
    return all_predictions


def predict_next_state(initial_features, causing_action_values, all_predictions):
    next_initial_features = []
    for causing_action in causing_action_values:
        prediction = all_predictions[causing_action]
        x_position, y_position = prediction['position']
        angle = prediction['angle']
        updated_components = update_components(initial_features, causing_action)
        next_initial_features.append(updated_components + [angle, x_position, y_position, causing_action])

    return next_initial_features


def calculate_distance_to_target(x1, y1, target_x, target_y):
    return np.sqrt((target_x - x1)**2 + (target_y - y1)**2)


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def update_component(components, param_type, channel, value):
    index_map = {'X': 0, 'Y': 1}
    if param_type == 'phase':
        components[index_map[channel]] = value
    else:  # For amplitude, adjust index by the number of phase components
        components[index_map[channel] + 2] = value
    return components
def update_components(initial_features, causing_action):
    components = initial_features[:4]
    param_type, channel, value = denormalize_action(causing_action)
    components = update_component(components, param_type, channel, value)
    return components

# Global list to accumulate positions
all_visited_positions = []
all_considered_positions = []
all_best_positions = []  # New global list to accumulate best positions

# Global variable to track if legends have been added
legends_added = False


def plot_results(visited_positions, considered_positions, best_positions_list, targets, threshold, initial_features):
    global legends_added  # Use the global variable

    if not visited_positions or not considered_positions:
        print("No positions visited.")
        return

    visited_x, visited_y = zip(*visited_positions)
    considered_x, considered_y = zip(*considered_positions)

    # Save the data for later use
    with open(f'plot_data_{time.time()}.pkl', 'wb') as f:
        pickle.dump(
            (visited_positions, considered_positions, best_positions_list, targets, threshold, initial_features), f)

    # Plot the positions
    plt.figure(figsize=(10, 10))
    plt.scatter(considered_x, considered_y, alpha=0.6, color='gray',
                label='Considered Positions' if not legends_added else "")  # Lighter plot for considered positions
    plt.scatter(visited_x, visited_y, alpha=1, color='blue',
                label='Visited Path' if not legends_added else "")  # Increased alpha for visited positions

    # Plot the best positions for each target
    for best_positions in best_positions_list:
        best_x, best_y = zip(*best_positions)  # Unpack best positions
        plt.scatter(best_x, best_y, alpha=1, color='green',
                    label='Best Path' if not legends_added else "")  # New scatter plot for best positions

    for target in targets:
        plt.plot(target[0], target[1], marker='*', color='red', markersize=15,
                 label='Target Position' if not legends_added else "")

    plt.plot(initial_features[5], initial_features[6], marker='s', color='black', markersize=15,
             label='Starting Position' if not legends_added else "")

    # Draw a circle around each target position with the threshold as the radius
    for target in targets:
        circle = plt.Circle((target[0], target[1]), threshold, color='red', fill=False, linestyle='--',
                            label='Threshold Circle' if not legends_added else "")
        plt.gca().add_artist(circle)

    # Set plot limits
    all_x = considered_x + visited_x + tuple(target[0] for target in targets) + (initial_features[5],)
    all_y = considered_y + visited_y + tuple(target[1] for target in targets) + (initial_features[6],)
    buffer = 0.1 + threshold
    plt.xlim(min(all_x) - buffer, max(all_x) + buffer)
    plt.ylim(min(all_y) - buffer, max(all_y) + buffer)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Exploration and Path to Targets')
    plt.grid(True)

    if not legends_added:
        plt.legend()  # Add a legend to distinguish the plots
        legends_added = True  # Update the global variable to indicate legends have been added

    # Save image with current timestamp
    plt.savefig(f'path_{time.time()}.png', dpi=600)
    plt.show()


# def a_star_search_segmental(initial_features, causing_action_values, target_x, target_y, max_levels=5, no_improvement_levels=1):
#     def predict_all_levels(current_features, previous_cost=float('inf'), level=0, no_improvement_count=0):
#         if level >= max_levels or no_improvement_count >= no_improvement_levels:
#             return [(current_features, [], [])]
#
#         all_paths = []
#         all_predictions = predict_next_state_for_all_actions(current_features, causing_action_values)
#
#         for causing_action, next_features in zip(causing_action_values, predict_next_state(current_features, causing_action_values, all_predictions)):
#             x, y = next_features[-3], next_features[-2]
#             distance_to_target = calculate_distance_to_target(x, y, target_x, target_y)
#
#             if distance_to_target >= previous_cost and level >= 4:  # Start the counter after the 2nd level
#                 new_no_improvement_count = no_improvement_count + 1
#             else:
#                 new_no_improvement_count = 0
#
#             paths = predict_all_levels(next_features, distance_to_target, level + 1, new_no_improvement_count)
#             for (final_features, path, positions) in paths:
#                 all_paths.append((final_features, [causing_action] + path, [(next_features[-3], next_features[-2])] + positions))
#
#         return all_paths
def a_star_search_segmental(initial_features, causing_action_values, target_x, target_y, max_levels=3, no_improvement_levels=3, top_paths=100, significant_distance=0.05):
    def predict_all_levels(current_features, previous_cost=float('inf'), level=0, no_improvement_count=0):
        if level >= max_levels or no_improvement_count >= no_improvement_levels:
            return [(current_features, [], [])]

        all_paths = []
        all_predictions = predict_next_state_for_all_actions(current_features, causing_action_values)

        # Store the paths and their distances to target
        path_distances = []
        for causing_action, next_features in zip(causing_action_values, predict_next_state(current_features, causing_action_values, all_predictions)):
            x, y = next_features[-3], next_features[-2]
            distance_to_target = calculate_distance_to_target(x, y, target_x, target_y)

            # If the distance change is significant enough, return the current path and positions immediately
            if distance_to_target <= significant_distance:
                return [(next_features, [causing_action], [(x, y)])]

            if distance_to_target >= previous_cost and level >= 3:  # Start the counter after the 2nd level
                new_no_improvement_count = no_improvement_count + 1
            else:
                new_no_improvement_count = 0

            path_distances.append((next_features, distance_to_target, causing_action, new_no_improvement_count))

        # Sort the paths based on their distances to target and select the top paths after 3rd level
        if level >= 3:
            path_distances.sort(key=lambda x: x[1])
            top_path_distances = path_distances[:top_paths]
        else:
            top_path_distances = path_distances

        # Explore the top paths
        for next_features, distance_to_target, causing_action, new_no_improvement_count in top_path_distances:
            paths = predict_all_levels(next_features, distance_to_target, level + 1, new_no_improvement_count)
            for (final_features, path, positions) in paths:
                all_paths.append((final_features, [causing_action] + path, [(next_features[-3], next_features[-2])] + positions))

        return all_paths

    def find_best_path(all_paths):
        best_final_features = None
        best_path = None
        best_positions = None
        best_cost = float('inf')

        for final_features, path, positions in all_paths:
            x, y = final_features[-3], final_features[-2]
            distance_to_target = calculate_distance_to_target(x, y, target_x, target_y)

            if distance_to_target < best_cost:
                best_cost = distance_to_target
                best_final_features = final_features
                best_path = path
                best_positions = positions

        return best_path, best_positions, best_cost, best_final_features

    all_paths = predict_all_levels(initial_features)
    best_path, best_positions, best_cost, best_final_features = find_best_path(all_paths)

    visited_positions = [(initial_features[-3], initial_features[-2])]  # Collect initial position
    considered_positions = []  # Track all considered positions

    for final_features, path, positions in all_paths:
        for causing_action in path:
            prediction = predict_next_state_for_all_actions(final_features, causing_action_values)[causing_action]
            x, y = prediction['position']
            considered_positions.append((x, y))
            visited_positions.append((x, y))

    # Append results to global lists
    all_visited_positions.extend(visited_positions)
    all_considered_positions.extend(considered_positions)
    all_best_positions.append(best_positions)

    return finalize_search(best_path, best_positions, [best_cost], best_final_features)

def finalize_search(path, best_positions, costs, final_features):
    # No need to call plot_results here
    return path, best_positions, costs, final_features




def generate_triangle_targets(current_x, current_y):
    targets = []
    for i in range(3):
        angle = i * (2 * np.pi) / 3
        target_x = current_x + 0.07 * np.cos(angle)
        target_y = current_y + 0.07 * np.sin(angle)
        targets.append((target_x, target_y))
    return targets

#write a function that takes in current x and y and generates 5 targets in a circle, taking the current x and y as the center
def generate_circle_targets(current_x, current_y):
    targets = []
    for i in range(3):
        angle = i * (2 * np.pi) / 3
        target_x = current_x + 0.3 * np.cos(angle)
        target_y = current_y + 0.3 * np.sin(angle)
        targets.append((target_x, target_y))
    return targets

if __name__ == "__main__":
    initial_features = [0.0, 0.0, 0.0, 0.0, 0.996383464798392, 0.738, 0.463, 0.0]

    # Generate triangle targets
    start_time = time.time()
    #triangle_targets = generate_triangle_targets(initial_features[5], initial_features[6])
    #circle_targets = generate_triangle_targets(initial_features[5], initial_features[6])
    #print("Triangle Targets to reach:", triangle_targets)
    circle_targets = [(0.33799999999999997, 0.463)]

    for target in circle_targets:
        print(f"Target to reach: {target}")
        reached_target = False
        current_initial_features = initial_features

        while not reached_target:
            # Call the A* search segmental function
            result_path, best_positions, costs, final_features = a_star_search_segmental(current_initial_features, causing_action_values, target[0], target[1])

            # Check the final position
            final_x, final_y = final_features[-3], final_features[-2]
            distance_to_target = calculate_distance_to_target(final_x, final_y, target[0], target[1])

            if distance_to_target <= 0.05:
                reached_target = True
            else:
                current_initial_features = final_features

        print("Path to target:", result_path)
        print("Number of actions:", len(result_path))
        print("Total cost:", sum(costs))
        print("Best positions are", best_positions)
        print("\n")
        print("Final features are", final_features)

        # Update initial_features for the next target
        #initial_features = final_features

    print("Time it took was", time.time() - start_time)

    # Plot all paths together after processing all targets
    plot_results(all_visited_positions, all_considered_positions, all_best_positions, circle_targets, 0.05, initial_features)
