import gym

from utils.FY8300 import SignalGenerator
from camera.camera import Camera
import cv2
import threading
import numpy as np
import h5py
import os
from datetime import datetime
from utils.predict_image import ResultVisualizer
import time
import torch
from gym.spaces import Box, Discrete
import math

# Constants
DEVICE = "cuda"
MODEL_PATH = "train_unet_model/results/Multiclass_2_model_12_13_23.pth"


class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()

        # Setup Action and Observation Spaces
        self.setup_spaces()
        self.cumulative_reward = 0.0
        self.low_level_done=False
        self.target = None

        self.prev_centers = []  # List to keep track of past centers
        self.max_prev_centers = 5
        self.components_initialized = False
        self.prev_phase_adjustments = None
        self.step_count = 0
        self.target_reached = True

        # Initialize Hardware and Software Components
        self.initialize_components()

        # Initialize Internal State Variables
        self.center_trajectory = []  # List to store center positions for trajectory
        self.initialize_state()
        self.current_episode = 0  # Initialize current_episode here
        self.current_step = (0)
        self.prev_angle = None  # Initialize previous angle
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sg_parameters = {
            'X': {'phase': 0, 'amplitude': 0},
            'Y': {'phase': 0, 'amplitude': 0},
            'Z': {'phase': 0, 'amplitude': 0},
        }

        # Create a unique folder for this experiment within RL_experiments_data based on the date and time
        self.folder_name = os.path.join("Results", self.current_time)
        os.makedirs(self.folder_name, exist_ok=True)

        # Create hdf5 path within the folder
        self.hdf5_file_path = os.path.join(self.folder_name, f"{self.current_time}_SAC_data.h5")

        self.set_physical_boundary()

        self.init_episode()  # Now you can call init_episode safely
        self.target_index = 0  # To keep track of the current target
        self.predefined_targets = self.generate_random_targets(20, 80)

    def setup_spaces(self):
        # 6 variables each with 2 possible values
        self.action_space = Discrete(8)  # 12 discrete actions

        # Define Observation Space
        self.observation_space = Box(low=0.0, high=1.0, shape=(7,),
                                     dtype=np.float32)  # Adjust according to actual state components

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

    def get_and_process_frame(self):
        frame = self.cam.get_latest_frame()


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
        return frame,  centers, angle, pred_bgr
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

    def generate_random_targets(self, num_targets=20, min_distance_to_boundary=80):
        targets = []
        safe_radius = self.circle_radius - min_distance_to_boundary
        for _ in range(num_targets):
            angle = 2 * np.pi * np.random.rand()  # Random angle between 0 and 2*pi
            distance = safe_radius * np.sqrt(np.random.rand())  # Random distance within the safe radius
            raw_target_x = self.circle_center[0] + distance * np.cos(angle)
            raw_target_y = self.circle_center[1] + distance * np.sin(angle)
            normalized_target_x = (raw_target_x - (self.circle_center[0] - self.circle_radius)) / (2 * self.circle_radius)
            normalized_target_y = (raw_target_y - (self.circle_center[1] - self.circle_radius)) / (2 * self.circle_radius)
            targets.append((normalized_target_x, normalized_target_y))
        return targets

    def generate_new_target(self):
        target = self.predefined_targets[self.target_index]
        self.target_index = (self.target_index + 1) % len(self.predefined_targets)  # Loop through the targets
        return target

    # Environment Logic
    def is_inside_boundary(self, center):
        cx, cy = center  # No need to flip the coordinates
        distance_to_center = np.linalg.norm(np.array([cx, cy]) - np.array(self.circle_center))

        return distance_to_center <= self.circle_radius

    def distance_to_boundary(self, normalized_center):
        # Denormalize the center coordinates
        denormalized_cx = (normalized_center[0] * 2 * self.circle_radius) + (self.circle_center[0] - self.circle_radius)
        denormalized_cy = (normalized_center[1] * 2 * self.circle_radius) + (self.circle_center[1] - self.circle_radius)
        denormalized_center = [denormalized_cx, denormalized_cy]

        # Calculate distance to center in the original scale
        distance_to_center = np.linalg.norm(np.array(denormalized_center) - np.array(self.circle_center))
        distance_to_boundary = self.circle_radius - distance_to_center

        # Normalize the distance to boundary
        normalized_distance_to_boundary = distance_to_boundary / self.circle_radius
        return normalized_distance_to_boundary



    def calculate_angle_difference(self, angle1, angle2):
        # Calculate the minimum difference between two angles
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)

    def calculate_reward(self, current_center, prev_distance_to_target):
        # Constants
        DISTANCE_REWARD_SCALE = 1.0  # Scaling factor for reward based on distance moved towards the target
        BACKWARD_MOVEMENT_PENALTY = 2.0  # Penalty for backward movement equal to distance moved backward
        CLOSER_REWARD_BOOST = 2.0  # Exponential reward boost factor for getting closer to the target

        # Calculate current distance to target (normalized)
        current_distance_to_target = np.linalg.norm(np.array(current_center) - np.array(self.target))

        # Calculate distance moved towards the target
        distance_moved_towards_target = prev_distance_to_target - current_distance_to_target

        # Initial reward based on moving closer to the target
        reward = distance_moved_towards_target * DISTANCE_REWARD_SCALE

        # Penalize backward movement
        if distance_moved_towards_target < 0:
            # Apply penalty for moving away from the target, equal to the negative distance moved (scaled)
            reward -= abs(distance_moved_towards_target) * BACKWARD_MOVEMENT_PENALTY
        else:
            # Apply an exponential boost to the reward for positive progress towards the target
            reward += np.exp(distance_moved_towards_target * CLOSER_REWARD_BOOST) - 1

        return reward

    def initialize_if_needed(self):
            if not self.components_initialized:
                self.initialize_components()



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



    def calculate_normalized_relative_angle(self, normalized_robot_center, robot_orientation, normalized_target):
        """
        Calculate the normalized relative angle from the robot's orientation to the target.

        :param normalized_robot_center: A tuple (x, y) representing the robot's center, normalized to [0, 1].
        :param robot_orientation: The current orientation angle of the robot in degrees.
        :param normalized_target: A tuple (x, y) representing the target's position, normalized to [0, 1].
        :return: The normalized relative angle to the target, scaled between 0 and 1.
        """
        if normalized_robot_center is None or normalized_target is None:
            return None

        # Calculate the angle to the target
        target_vector_x = normalized_target[0] - normalized_robot_center[0]
        target_vector_y = normalized_target[1] - normalized_robot_center[1]
        angle_to_target = math.degrees(math.atan2(target_vector_y, target_vector_x))

        # Calculate the relative angle
        relative_angle = angle_to_target - robot_orientation

        # Normalize the relative angle to the range [0, 360]
        normalized_angle = relative_angle % 360

        # Scale the normalized angle to the range [0, 1]
        normalized_relative_angle = normalized_angle / 360

        return normalized_relative_angle

    def reset(self):
        print(f"low level reset called at low level step {self.current_step}")

        self.initialize_if_needed()
        self.current_step = 0
        self.current_episode += 1
        self.low_level_done = False

        self.cumulative_reward = 0.0
        self.center_trajectory = []

        # Initialize phase and amplitude values
        self.set_sg_parameters('X', 'phase', 0)
        self.set_sg_parameters('Y', 'phase', 0)
        self.set_sg_parameters('X', 'amplitude', 0)
        self.set_sg_parameters('Y', 'amplitude', 0)


        # Capture the initial phase and amplitude values
        prev_phase_values = [self.sg_parameters[ch]['phase'] / 180 for ch in ['X', 'Y']]
        prev_amplitude_values = [self.sg_parameters[ch]['amplitude'] / 7 for ch in ['X', 'Y']]
        while True:
            frame, centers, angle, pred_bgr = self.get_and_process_frame()
            front_center = centers.get(1, None)
            back_center = centers.get(2, None)
            current_center = self.calculate_current_center(front_center, back_center)

            # Check if current center is found and angle is not zero
            if current_center is None or angle is None:
                action_required = "Current center not found" if current_center is None else "Angle is zero"
                user_input = input(f"{action_required}. Please adjust the robot and press 'Y' to continue: ")
                while user_input.lower() != 'y':
                    user_input = input("Press 'Y' after repositioning the robot: ")
            else:
                break

        normalized_angle = angle / 360.0

        # Calculate the distance to the boundary
        if current_center is not None:

            distance_to_boundary = self.distance_to_boundary(current_center)
        else:
            distance_to_boundary = np.inf  # Use a large number or some other default


        # User intervention for repositioning the robot
        normalized_boundary_threshold = 40 / self.circle_radius  # Normalized threshold for user intervention
        if distance_to_boundary <= normalized_boundary_threshold:
            user_input = input("Robot too close to boundary. Reposition robot and press 'Y' to continue: ")
            while user_input.lower() != 'y':
                user_input = input("Press 'Y' after repositioning the robot: ")
                frame, centers, angle, pred_bgr = self.get_and_process_frame()
                front_center = centers.get(1, None)
                back_center = centers.get(2, None)
                current_center = self.calculate_current_center(front_center, back_center)

        # Set a new target if the low level is done or it's the start of a new episode
        if self.target_reached or self.current_episode == 1:
            self.target = self.generate_new_target()
            self.target_reached = False  # Reset the flag

        if current_center is not None and self.target is not None:
            raw_distance_to_target = np.linalg.norm(np.array(current_center) - np.array(self.target))
        else:
            normalized_distance_to_target = np.inf  # Or some suitable default value

        self.previous_distance_to_target = raw_distance_to_target


        # Angle to target (considering normalization)
        angle = normalized_angle * 360  # Convert back to degrees

        target_vector = self.calculate_normalized_relative_angle(current_center, angle, self.target)

        # Flatten the state into a single array
        flattened_state = np.concatenate([
            np.round(np.array([normalized_angle], dtype=np.float32), 3),
            np.round(np.array([raw_distance_to_target], dtype=np.float32), 3),
            np.round(np.array([target_vector], dtype=np.float32), 3),
            np.round(np.array(prev_phase_values, dtype=np.float32), 3),  # Include the captured phase values
            np.round(np.array(prev_amplitude_values, dtype=np.float32), 3)  # Include the captured amplitude values
        ])

        self.prev_angle = normalized_angle
        self.prev_center = current_center
        return flattened_state

    def init_episode(self):
        self.current_episode += 1  # Increment the episode number
        self.current_step = 0  # Reset the step count for the new episode


    def save_to_hdf5(
            self,
            hdf5_file_path,
            current_episode,
            current_step,
            current_center,
            reward,
            done,
            info_reason,
            frame,
            pred_bgr,
            prev_phase_values,  # New parameter
            prev_amplitude_values,  # New parameter
            centers,
            angle,
            distance_to_boundary,
            target,
            circle_radius,
            target_vector,
            raw_distance_to_target,
            action
    ):
        with h5py.File(hdf5_file_path, "a") as hf:
            # Check if the episode group already exists, create it if not
            episode_grp_name = f"episode_{current_episode}"
            if episode_grp_name in hf:
                episode_grp = hf[episode_grp_name]
            else:
                episode_grp = hf.create_group(episode_grp_name)

            # Now create a group for the current step within the episode group
            step_grp = episode_grp.create_group(f"step_{current_step}")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
                           :-3
                           ]  # Millisecond precision
            step_grp.create_dataset("timestamp", data=current_time)
            step_grp.create_dataset("current_center", data=current_center)
            step_grp.create_dataset("reward", data=reward)
            step_grp.create_dataset("done", data=done)
            step_grp.create_dataset("info_reason", data=str(info_reason))
            step_grp.create_dataset("frame", data=frame)
            step_grp.create_dataset("pred_gbr", data=pred_bgr)
            step_grp.create_dataset("prev_phase_values", data=prev_phase_values)
            step_grp.create_dataset("prev_amplitude_values", data=prev_amplitude_values)
            step_grp.create_dataset("centers", data=str(centers))
            step_grp.create_dataset("angle", data=angle)
            step_grp.create_dataset("distance_to_boundary", data=distance_to_boundary)
            step_grp.create_dataset("target", data=target)
            step_grp.create_dataset("circle_radius", data=circle_radius)
            step_grp.create_dataset("target_vector", data=target_vector)
            step_grp.create_dataset("raw_distance_to_target", data=raw_distance_to_target)
            step_grp.create_dataset("action", data=action)  # New dataset for action


    def save_to_hdf5_thread(self, current_center, reward, done, info, frame, pred_bgr, phase_adjustments,
                            amplitude_adjustments, centers, angle, distance_to_boundary, target_vector,
                            raw_distance_to_target, action):  # New parameter
        threading.Thread(
            target=self.save_to_hdf5,
            args=(
                self.hdf5_file_path,
                self.current_episode,
                self.current_step,
                current_center,
                reward,
                done,
                info.get("reason", ""),
                frame,
                pred_bgr,
                phase_adjustments,
                amplitude_adjustments,
                centers,
                angle,
                distance_to_boundary,
                self.target,
                self.circle_radius,
                target_vector,
                raw_distance_to_target,
                action

            ),
        ).start()

    def check_robot_state(self, current_center, angle):
        done = False
        info = {}
        if angle is None:
            normalized_angle = 0.0
            done = True
            self.low_level_done = True
        else:
            normalized_angle = angle / 360.0

        if current_center is not None:
            distance_to_boundary = self.distance_to_boundary(current_center)
            normalized_boundary_threshold = 40 / self.circle_radius
            if distance_to_boundary <= normalized_boundary_threshold:
                self.low_level_done = True
                done = True
                self.target_reached = True

        raw_distance_to_target = np.linalg.norm(np.array(current_center) - np.array(self.target))
        if raw_distance_to_target <= (30 / (2 * self.circle_radius)):
            self.low_level_done = True
            done = True
            info["termination_reason"] = "Target reached"
            self.target_reached = True
            print("Target reached")

        return normalized_angle, done, info

    def set_sg_parameters(self, channel, param_type, value):
        if param_type not in ['phase', 'amplitude']:
            raise ValueError("Parameter type must be 'phase' or 'amplitude'.")
        if channel not in self.sg_parameters:
            raise ValueError("Invalid channel. Must be 'X', 'Y'")

        self.sg_parameters[channel][param_type] = value

        # Map 'X', 'Y', 'Z' to 1, 2, 3 for hardware interaction
        channel_mapping = {'X': 1, 'Y': 2}
        hardware_channel = channel_mapping[channel]

        # sg controls the signal generator
        self.sg.set_parameter(hardware_channel, param_type, f"{value}{'ampere' if param_type == 'amplitude' else ''}")
        #apply channel 3 to 7 ampere


    def step(self, action):
        frame,centers, angle, pred_bgr = self.get_and_process_frame()
        front_center = centers.get(1, None)
        back_center = centers.get(2, None)
        current_center = self.calculate_current_center(front_center, back_center)
        normalized_angle, done, info = self.check_robot_state(current_center, angle)

        #we need to calculate distance to target before we apply the action
        raw_distance_to_target = np.linalg.norm(np.array(current_center) - np.array(self.target))

        # Capture the current phase and amplitude values before the action is applied
        prev_phase_values = [self.sg_parameters[ch]['phase'] / 180 for ch in ['X', 'Y']]
        prev_amplitude_values = [self.sg_parameters[ch]['amplitude'] / 7 for ch in ['X', 'Y']]

        if action < 4:  # Phase adjustments
            channel_index = action // 2
            phase_value = [0, 180][action % 2]
            param_type = 'phase'
        else:  # Amplitude adjustments
            channel_index = (action - 4) // 2
            amplitude_value = [0, 7][(action - 4) % 2]
            param_type = 'amplitude'

        channel = ['X', 'Y'][channel_index]
        value = phase_value if param_type == 'phase' else amplitude_value

        # Set the parameter
        self.set_sg_parameters(channel, param_type, value)


        reward = self.calculate_reward(current_center,prev_distance_to_target=self.previous_distance_to_target)
        self.previous_distance_to_target = raw_distance_to_target
        self.cumulative_reward += reward
        self.current_step += 1
        target_vector = self.calculate_normalized_relative_angle(current_center, angle, self.target)

        self.save_to_hdf5_thread(current_center, reward, done, info, frame, pred_bgr, prev_phase_values,
                                 prev_amplitude_values,
                                  centers, angle,
                                 self.distance_to_boundary(current_center), target_vector,
                                 np.linalg.norm(np.array(current_center) - np.array(self.target)),
                                 action)

        self.visualize_frame(frame, centers, angle, reward, current_center)

        flattened_state = np.concatenate([
            np.round(np.array([normalized_angle], dtype=np.float32), 3),
            np.round(np.array([np.linalg.norm(np.array(current_center) - np.array(self.target))], dtype=np.float32), 3),
            np.round(np.array([target_vector], dtype=np.float32), 3),
            np.round(np.array(prev_phase_values, dtype=np.float32), 3),  # Use the captured phase values
            np.round(np.array(prev_amplitude_values, dtype=np.float32), 3)  # Use the captured amplitude values
        ])

        self.prev_angle = normalized_angle
        self.prev_center = current_center

        return flattened_state, reward, done, info

    def visualize_frame(self, frame, centers, angle, reward, current_center):


        front_center = centers.get(1, None)
        back_center = centers.get(2, None)
        # Calculate penalty for boundary proximity
        denormalized_current_center = [
            (coord * 2 * self.circle_radius) + (self.circle_center[i] - self.circle_radius)
            for i, coord in enumerate(current_center)]
        BOUNDARY_THRESHOLD=60/self.circle_radius
        denormalized_target = [(coord * 2 * self.circle_radius) + (self.circle_center[i] - self.circle_radius) for
                               i, coord in enumerate(self.target)]

        # Draw the physical boundary
        if hasattr(self, 'circle_center') and hasattr(self, 'circle_radius'):
            cv2.circle(frame, self.circle_center, self.circle_radius, (255, 255, 255), 2)

        # Visualize the distance to boundary
        if denormalized_current_center is not None and hasattr(self, 'circle_center'):
            vec_to_center = np.array(self.circle_center) - np.array(denormalized_current_center)
            vec_to_center = vec_to_center / np.linalg.norm(vec_to_center)
            point_on_boundary = np.array(self.circle_center) - vec_to_center * self.circle_radius
            cv2.line(frame, tuple(map(int, denormalized_current_center)), tuple(map(int, point_on_boundary)), (255, 0, 255), 2)
            distance_to_boundary = self.distance_to_boundary(current_center)
            if distance_to_boundary < BOUNDARY_THRESHOLD:
                penalty = BOUNDARY_THRESHOLD - distance_to_boundary
                text_position = tuple(map(int, (denormalized_current_center + point_on_boundary) / 2))
                cv2.putText(frame, f"Penalty: {-penalty:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        cv2.imshow("Debug Frame", frame)
        cv2.waitKey(1)

        if denormalized_current_center is not None:
            self.center_trajectory.append(denormalized_current_center)
            arrow_length = 50  # Length of the arrow in pixels

            if angle is not None:
                angle_rad_corrected = -np.radians(angle)
                arrow_end = (int(denormalized_current_center[0] + arrow_length * np.cos(angle_rad_corrected)),
                             int(denormalized_current_center[1] + arrow_length * np.sin(angle_rad_corrected)),)

                if front_center is not None and back_center is not None:
                    cv2.arrowedLine(
                        frame,
                        tuple(map(int, back_center)),
                        tuple(map(int, front_center)),
                        (0, 0, 255),
                        1,
                        tipLength=0.2,
                    )
                    cv2.putText(
                        frame,
                        f"Angle: {angle:.2f} degrees",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        1,
                    )
            else:
                print("Warning: Not drawing arrow because angle is None.")

        for center in self.center_trajectory[::40]:
            cv2.circle(frame, tuple(map(int, center)), 3, (255, 255, 0), -1)

        if not np.all(denormalized_current_center == [0, 0]):
            cv2.circle(frame, tuple(map(int, denormalized_current_center)), 3, (0, 255, 0), -1)

        # Visualize the target and line to the target
        if denormalized_target is not None and denormalized_current_center is not None:
            cv2.circle(frame, tuple(map(int, denormalized_target)), 25, (0, 255, 0), 2)  # Draw target circle
            cv2.line(frame, tuple(map(int, denormalized_current_center)), tuple(map(int, denormalized_target)), (255, 0, 0),
                     2)  # Draw line to target

            # Display the reward
            text_position = (
            int((denormalized_current_center[0] + denormalized_target[0]) / 2), int((denormalized_current_center[1] + denormalized_target[1]) / 2))
            cv2.putText(frame, f"Reward: {reward:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        cv2.imshow("Debug Frame", frame)
        cv2.waitKey(1)
    def render(self, mode="human"):
        frame = self.cam.get_latest_frame()
        frame_resized, centers, angle, pred_bgr, orientation_labels = self.result_visualizer.process_frame(
            self.result_visualizer.model, frame, self.result_visualizer.device, self.result_visualizer.color_map
        )

    def close(self):
        if hasattr(self, "cam_thread"):
            self.cam_thread.join()


