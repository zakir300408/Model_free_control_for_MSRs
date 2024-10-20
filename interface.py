import time
from FY8300 import SignalGenerator
from camera.camera import Camera
import cv2
import threading
import numpy as np
import os
from datetime import datetime
from predict_image import ResultVisualizer
import torch
from predic_action import *
import h5py
import threading

# Constants
DEVICE = "cuda"
MODEL_PATH = "Multiclass_2_model_12_13_23.pth"


class RobotEnv():
    def __init__(self):
        super(RobotEnv, self).__init__()
        self.normalized_causing_action = 0
        self.target_x = 0
        self.target_y = 0
        self.components_initialized = False
        # Initialize Hardware and Software Components
        self.initialize_components()
        self.initialize_state()
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sg_parameters = {
            'X': {'phase': 0, 'amplitude': 0},
            'Y': {'phase': 0, 'amplitude': 0},
            'Z': {'phase': 0, 'amplitude': 0},
        }

        # Create a unique folder for this experiment within RL_experiments_data based on the date and time
        self.folder_name = os.path.join("Interface_control_data", self.current_time)
        os.makedirs(self.folder_name, exist_ok=True)

        # Create hdf5 path within the folder
        self.hdf5_file_path = os.path.join(self.folder_name, f"{self.current_time}_Experiment_data.h5")

        self.set_physical_boundary()

        #episode counter
        self.episode = 0
        self.step_counter = 0
        self.prev_current_center_x = 0
        self.prev_current_center_y = 0
        self.normalized_prev_angle = 0
        self.prev_phase_value_x = 0
        self.prev_phase_value_y = 0
        self.prev_phase_value_z = 0
        self.prev_amplitude_value_x = 0
        self.prev_amplitude_value_y = 0
        self.prev_amplitude_value_z = 0

    def initialize_components(self):
        if not self.components_initialized:
            # Initialize Signal Generator
            self.sg = SignalGenerator()
            self.sg.initialize_channel(1, 1, 2)
            self.sg.initialize_channel(2, 1, 2)
            self.sg.initialize_channel(3, 1, 2)
            self.sg.first_init()
            self.sg.set_parameter(3, 'amplitude', '7 ampere')
            time.sleep(10)
            # Load the model first
            model = torch.load(MODEL_PATH)
            # Initialize ResultVisualizer and Camera
            self.result_visualizer = ResultVisualizer(model, DEVICE)
            self.cam = Camera()  # Create a Camera object
            self.cam_thread = threading.Thread(target=self.cam.capture_video)
            self.cam_thread.start()
            self.components_initialized = True

    def initialize_state(self):
        self.prev_center = None
        self.staying_counter = 0
        self.previous_distance_to_target = None
        self.boundary_coordinates = None

    # User Input Methods
    def set_physical_boundary(self):
        frame = self.cam.get_latest_frame()

        # Calculate the center of the frame
        frame_height, frame_width = frame.shape[:2]
        self.circle_center = (frame_width // 2, frame_height // 2)
        self.circle_radius = 250

        # Draw the circular boundary
        self.frame_with_boundary = cv2.circle(
            frame.copy(),
            self.circle_center,
            self.circle_radius,
            (0, 255, 0),
            2,
        )

        # Calculate the max_distance based on the physical boundary
        self.max_distance = self.circle_radius * np.sqrt(2)


    def calculate_angle_difference(self, angle1, angle2):
        # Calculate the minimum difference between two angles
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)


    def initialize_if_needed(self):
            if not self.components_initialized:
                self.initialize_components()

    def get_and_process_frame(self):
        frame = self.cam.get_latest_frame()
        original_frame = frame.copy()
        (
            frame_resized,
            centers,
            angle,
            pred_bgr,
            orientation_labels,
        ) = self.result_visualizer.process_frame(
            frame,
            self.result_visualizer.device,
            self.result_visualizer.color_map,
        )
        return frame, original_frame, centers, angle, pred_bgr

    def calculate_current_center(self, front_center, back_center):
        if front_center is not None and back_center is not None:
            raw_center_x = (front_center[0] + back_center[0]) / 2
            raw_center_y = (front_center[1] + back_center[1]) / 2

            # Normalize the coordinates to range [0, 1]
            normalized_center_x = (raw_center_x - (self.circle_center[0] - self.circle_radius)) / (
                        2 * self.circle_radius)
            normalized_center_y = (raw_center_y - (self.circle_center[1] - self.circle_radius)) / (
                        2 * self.circle_radius)

            return [normalized_center_x, normalized_center_y]
        else:
            return None

    def set_sg_parameters(self, channel, param_type, value):
        if param_type not in ['phase', 'amplitude']:
            raise ValueError("Parameter type must be 'phase' or 'amplitude'.")
        if channel not in self.sg_parameters:
            raise ValueError("Invalid channel. Must be 'X', 'Y'")

        self.sg_parameters[channel][param_type] = value

        # Map 'X', 'Y', 'Z' to 1, 2, 3 for hardware interaction
        channel_mapping = {'X': 1, 'Y': 2}
        hardware_channel = channel_mapping[channel]

        # Sg controls actual hardware
        self.sg.set_parameter(hardware_channel, param_type, f"{value}{'ampere' if param_type == 'amplitude' else ''}")

    def write_hdf5(self, hdf5_file_path, data):
        with h5py.File(hdf5_file_path, "a") as hf:
            episode_grp_name = f"episode_1"
            if episode_grp_name in hf:
                episode_grp = hf[episode_grp_name]
            else:
                episode_grp = hf.create_group(episode_grp_name)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            step_grp = episode_grp.create_group(f"step_{data['step_counter']}")
            step_grp.create_dataset("current_time", data=current_time)
            step_grp.create_dataset("current_center_x", data=data['current_center_x'])
            step_grp.create_dataset("current_center_y", data=data['current_center_y'])
            step_grp.create_dataset("angle", data=data['angle'])
            step_grp.create_dataset("prev_phase_value_x", data=data['prev_phase_values'][0])
            step_grp.create_dataset("prev_phase_value_y", data=data['prev_phase_values'][1])
            step_grp.create_dataset("prev_amplitude_value_x", data=data['prev_amplitude_values'][0])
            step_grp.create_dataset("prev_amplitude_value_y", data=data['prev_amplitude_values'][1])
            step_grp.create_dataset("current_action", data=data['current_action'])
            step_grp.create_dataset("current_center_x_denormalized", data=data['current_center_x_denormalized'])
            step_grp.create_dataset("current_center_y_denormalized", data=data['current_center_y_denormalized'])
            step_grp.create_dataset("circle_radius", data=data['circle_radius'])
            step_grp.create_dataset("frame", data=data['frame'])
            step_grp.create_dataset("pred_bgr", data=data['pred_bgr'])
            step_grp.create_dataset("max_distance", data=data['max_distance'])
            step_grp.create_dataset("normalized_causing_action", data=data['normalized_causing_action'])
            #circle center
            step_grp.create_dataset("circle_center_x", data=self.circle_center[0])
            step_grp.create_dataset("circle_center_y", data=self.circle_center[1])

    def step(self, action):
        global phase_value, amplitude_value
        start_time = time.time()
        self.initialize_if_needed()
        frame, original_frame, centers, angle, pred_bgr = self.get_and_process_frame()
        front_center = centers.get(1, None)
        back_center = centers.get(2, None)
        current_center = self.calculate_current_center(front_center, back_center)

        current_center_x = current_center[0]
        current_center_y = current_center[1]
        current_center_x_denormalized = current_center[0] * 2 * self.circle_radius + (
                self.circle_center[0] - self.circle_radius)
        current_center_y_denormalized = current_center[1] * 2 * self.circle_radius + (
                self.circle_center[1] - self.circle_radius)

        # Start a new thread for writing the HDF5 file
        data = {
            'step_counter': self.step_counter,
            'current_center_x': current_center_x,
            'current_center_y': current_center_y,
            'angle': angle,
            'prev_phase_values': [self.sg_parameters[ch]['phase'] / 180 for ch in ['X', 'Y']],
            'prev_amplitude_values': [self.sg_parameters[ch]['amplitude'] / 7 for ch in ['X', 'Y']],
            'current_action': action,
            'current_center_x_denormalized': current_center_x_denormalized,
            'current_center_y_denormalized': current_center_y_denormalized,
            'circle_radius': self.circle_radius,
            'frame': frame,
            'pred_bgr': pred_bgr,
            'max_distance': self.max_distance,
            'normalized_causing_action': action / 7
        }

        threading.Thread(target=self.write_hdf5, args=(self.hdf5_file_path, data)).start()

        # Apply the action
        if action < 4:
            channel_index = action // 2
            phase_value = [0, 180][action % 2]
            param_type = 'phase'
        else:
            channel_index = (action - 4) // 2
            amplitude_value = [0, 7][(action - 4) % 2]
            param_type = 'amplitude'

        channel = ['X', 'Y'][channel_index]
        value = phase_value if param_type == 'phase' else amplitude_value
        self.set_sg_parameters(channel, param_type, value)

        # Update the parameters after the action is applied
        updated_phase_values = [self.sg_parameters[ch]['phase'] / 180 for ch in ['X', 'Y']]
        updated_amplitude_values = [self.sg_parameters[ch]['amplitude'] / 7 for ch in ['X', 'Y']]

        self.visualize_frame(frame, centers, angle, (current_center_x_denormalized, current_center_y_denormalized))

        self.prev_phase_value_x = updated_phase_values[0]
        self.prev_phase_value_y = updated_phase_values[1]
        self.prev_amplitude_value_x = updated_amplitude_values[0]
        self.prev_amplitude_value_y = updated_amplitude_values[1]
        self.prev_current_center_x = current_center_x
        self.prev_current_center_y = current_center_y
        self.prev_angle = angle
        self.normalized_prev_angle = self.prev_angle / 360
        self.normalized_causing_action = action / 7
        print("normalized causing action is", self.normalized_causing_action)

        initial_features = [self.prev_phase_value_x, self.prev_phase_value_y,
                            self.prev_amplitude_value_x, self.prev_amplitude_value_y,
                             self.normalized_prev_angle, self.prev_current_center_x,
                            self.prev_current_center_y, self.normalized_causing_action]

        self.step_counter += 1
        print("current position is", current_center_x, current_center_y)
        return initial_features

    def visualize_frame(self, frame, centers, angle, current_center, tolerance=0.07):

        # Draw the circular boundary
        frame_with_boundary = cv2.circle(
            frame.copy(),
            self.circle_center,
            self.circle_radius,
            (0, 255, 0),
            2,
        )

        # Draw the target location as a star
        target_location = (int(self.target_x * frame.shape[1]), int(self.target_y * frame.shape[0]))
        cv2.drawMarker(frame_with_boundary, target_location, (0, 0, 255), cv2.MARKER_STAR, 10)

        # Draw a threshold circle of radius r around the target location
        threshold_radius = int(tolerance * frame.shape[1])  # Calculate the radius as a percentage of the frame's width
        cv2.circle(frame_with_boundary, target_location, threshold_radius, (0, 0, 255), 2)

        # Draw the current center
        if current_center is not None:
            cv2.circle(frame_with_boundary, tuple(map(int, current_center)), 5, (0, 255, 0), -1)

        # Draw the centers of the detected objects
        for center in centers.values():
            cv2.circle(frame_with_boundary, center, 5, (0, 0, 255), -1)

        # Draw the angle
        if angle is not None:
            angle_text = f"Angle: {angle:.2f}"
            cv2.putText(frame_with_boundary, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Frame", frame_with_boundary)
        cv2.waitKey(1)

    def reset(self):
        self.sg.set_parameter(1, 'amplitude', 0)
        self.sg.set_parameter(2, 'amplitude', 0)
        self.prev_amplitude_value_x = 0
        self.prev_amplitude_value_y = 0
        return

    def close(self):
        if hasattr(self, "cam_thread"):
            self.cam_thread.join()
        cv2.destroyAllWindows()
        self.sg.close()
        self.cam.release()

# Global list to accumulate positions
all_visited_positions = []
all_considered_positions = []
all_best_positions = []  # New global list to accumulate best positions

# def generate_triangle_targets(current_x, current_y):
#     targets = []
#     for i in range(3):
#         angle = i * (2 * np.pi) / 3
#         target_x = current_x * np.cos(angle)
#         target_y = current_y + 0.2 * np.sin(angle)
#         targets.append((target_x, target_y))
#     return targets

def generate_target(current_x, current_y):
    targets=[]
    target_x = current_x + 0.1
    target_y = current_y + 0.5
    targets.append((target_x, target_y))
    return targets

def generate_square_targets(current_x, current_y, side_length=0.3):
    targets = [
        (current_x, current_y),  # Start at the current position of the robot
        (current_x + side_length, current_y),  # Move right
        (current_x + side_length, current_y - side_length),  # Move up (decreasing y)
        (current_x, current_y - side_length)  # Move left to complete the square
    ]
    return targets

#function to generate two targets, one is x-0.6, y=y, and other is back to the original position
def generate_two_targets(current_x, current_y):
    targets = [
        (current_x - 0.4, current_y)  # Move left
          # Move right to complete the square
    ]
    return targets
def generate_triangle_targets(current_x, current_y):
    targets = [
        (current_x, current_y),  # Start at the current position of the robot
        (current_x + 0.5, current_y),  # Move right
        (current_x, current_y - 0.3)  # Move up
    ]
    return targets




if __name__ == "__main__":
    env = RobotEnv()

    print("Environment initialized")
    # Get the initial features from the first step
    initial_features = env.step(0)
    print("Initial features:", initial_features)

    #triangle_targets = generate_square_targets(initial_features[5], initial_features[6])
    targets=generate_two_targets(initial_features[5], initial_features[6])

    start_time = time.time()
    for target in targets:
        print(f"Target to reach: {target}")
        target_reached = False
        while not target_reached:
            # Update the target position in the initial features
            result_path, best_positions, costs, final_features = a_star_search_segmental(initial_features, causing_action_values, target[0], target[1])
            print("Path to target:", result_path)
            print("Number of actions:", len(result_path))
            # Now let's apply the actions to the environment
            for action in result_path:
                action = round(action * 7)
                print("Action to apply:", action)
                initial_features = env.step(action)
                time.sleep(2)
                print("Action applied:", action)
                print("New features:", initial_features)
                # Append the visited positions to the all_visited_positions list
                all_visited_positions.append((initial_features[5], initial_features[6]))
                # Append the considered positions to the all_considered_positions list
                all_considered_positions.append((initial_features[5], initial_features[6]))

            all_best_positions.append(best_positions)  # Append the best_positions to the all_best_positions list

            # Check if the position is within 0.05 of the target
            current_position = (initial_features[5], initial_features[6])
            if abs(current_position[0] - target[0]) < 0.05 and abs(current_position[1] - target[1]) < 0.05:
                target_reached = True
                print("Target reached!")
            else:
                print("Resetting environment for further actions...")
                # Reset amplitude values in initial_features
                env.sg.set_parameter(1, 'amplitude', 0)
                env.sg.set_parameter(2, 'amplitude', 0)
                initial_features[2] = 0
                initial_features[3] = 0
                #now print the updated initial features supplied to next step
                print("Updated features:", initial_features)

    print("Time it took was ", time.time() - start_time)

    # Plot all paths together after processing all targets
    plot_results(all_visited_positions, all_considered_positions, all_best_positions, targets, 0.05, initial_features)




