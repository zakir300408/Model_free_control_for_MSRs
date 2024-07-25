import logging
import warnings
import os
import numpy as np
import pickle
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from New_lowlevel import RobotEnv

class BufferSavingCallback(BaseCallback):
    def __init__(self, model_save_path, checkpoint_freq=5000, verbose=0):
        super(BufferSavingCallback, self).__init__(verbose)
        self.model_save_path = model_save_path
        self.checkpoint_freq = checkpoint_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.checkpoint_freq == 0:
            # Saving the model
            model_save_path = os.path.join(self.model_save_path, f"ppo_model_step_new_PPO_{self.num_timesteps}.zip")
            self.model.save(model_save_path)

            # Saving the random state
            save_random_state(self.model_save_path, self.num_timesteps)

            if self.verbose > 0:
                logging.info(f"Checkpoint saved at step {self.num_timesteps}")

        return True

def load_model_and_buffer(model_save_path, buffer_save_path, step):
    try:
        model = PPO.load(os.path.join(model_save_path, f"ppo_model_step_new_PPO_{step}.zip"))
        buffer_save_file = os.path.join(buffer_save_path, f"replay_buffer_new_PPO_{step}.pkl")
        with open(buffer_save_file, 'rb') as f:
            replay_buffer = pickle.load(f)
            model.replay_buffer = replay_buffer
        load_random_state(model_save_path, step)
        logging.info(f"Model and buffer loaded from step {step}.")
        return model
    except FileNotFoundError:
        logging.warning("Model and buffer files not found. Starting from scratch.")
        return None
def save_random_state(save_path, step):
    random_state = np.random.get_state()
    with open(os.path.join(save_path, f"random_state_new_PPP_{step}.pkl"), 'wb') as f:
        pickle.dump(random_state, f)

def load_random_state(load_path, step):
    try:
        with open(os.path.join(load_path, f"random_state_new_PPP_{step}.pkl"), 'rb') as f:
            random_state = pickle.load(f)
            np.random.set_state(random_state)
    except FileNotFoundError:
        logging.warning("Random state file not found. Continuing with a new random state.")

model_save_path = "saved_models"
buffer_save_path = os.path.join(model_save_path, "replay_buffers")
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(buffer_save_path):
    os.makedirs(buffer_save_path)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize the environment
low_level_env = RobotEnv()
vec_env = DummyVecEnv([lambda: low_level_env])
logging.info("Initialized low-level environment.")

def linear_schedule(initial_value, final_value):
    """
    Create a schedule for a linearly decreasing learning rate.

    :param initial_value: The initial learning rate.
    :param final_value: The final learning rate.
    :return: A function that takes the current progress and returns the learning rate.
    """
    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0 (end)
        """
        return progress * (initial_value - final_value) + final_value

    return func

# Define initial and final learning rates
initial_lr = 0.003
final_lr = 0.009

# Create the learning rate scheduler
learning_rate_schedule = linear_schedule(initial_lr, final_lr)

# Define network parameters for SAC
policy_kwargs = dict(net_arch=[30, 30])  # Adjusted to a slightly smaller network

last_checkpoint_step = 1200200
model = load_model_and_buffer(model_save_path, buffer_save_path, last_checkpoint_step)

if model is None:
    # Adjusted Entropy Coefficient for more exploration
    ent_coef = 0.6  # Increased from 0.2 to 0.4 to enhance exploration


    def dynamic_lr_schedule(step):
        """
        Adjusts the learning rate dynamically based on the training step.
        Introduces more variance in learning rate to promote exploration.
        """
        if step < 10000:
            return 0.003  # Higher learning rate initially for more exploration
        elif step < 20000:
            return 0.002  # Reduce as model starts converging
        else:
            return 0.001  # Fine-tuning the policy


    # Initialize PPO model with adjusted entropy and dynamic learning rate
    model = PPO(
        MlpPolicy,
        vec_env,
        learning_rate=dynamic_lr_schedule,  # Using the dynamic learning rate schedule
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo_low_level_tensorboard/",
        clip_range=0.2,
        n_steps=128,
        batch_size=128,
        n_epochs=10,
        ent_coef=ent_coef,  # Adjusted entropy coefficient
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        sde_sample_freq=-1,
        target_kl=None,
        device='auto',
        _init_setup_model=True
    )

else:
    model.set_env(vec_env)


callback = BufferSavingCallback(model_save_path, verbose=1)

model.learn(total_timesteps=50000, log_interval=10, callback=callback)


model.save(os.path.join(model_save_path, "PPO_low_level_model"))
logging.info("Training completed and model saved.")
