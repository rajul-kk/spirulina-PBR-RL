import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from world_model import WorldModel
from reward_functions import default_reward_function

class DataHandler:
    """
    Handles data normalization and cleaning for the Discrete PBR environment.
    """
    def __init__(self):
        # --- OBSERVATION BOUNDS (Density) ---
        # Assuming density ranges from 0.0 to 1.0
        self.obs_low = np.array([0.0], dtype=np.float32)
        self.obs_high = np.array([1.0], dtype=np.float32)

    def normalize_observation(self, obs):
        """
        Normalizes physical observation to [0, 1] range.
        Note: If obs is already density 0-1, this just clips it.
        """
        obs = np.clip(obs, self.obs_low, self.obs_high)
        normalized = (obs - self.obs_low) / (self.obs_high - self.obs_low)
        return normalized.astype(np.float32)

    def clean_observation(self, obs):
        """
        Handle NaNs or invalid values.
        """
        if np.any(np.isnan(obs)):
            obs = np.nan_to_num(obs, nan=0.0)
        return obs

class PhotobioreactorDiscreteEnv(gym.Env):
    """
    Custom Gymnasium Environment for a simulated Photobioreactor (Discrete).
    Uses a data-driven World Model for transitions.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, model_path="world_model.pth", reward_func=None):
        super(PhotobioreactorDiscreteEnv, self).__init__()
        
        self.data_handler = DataHandler()
        self.reward_func = reward_func if reward_func is not None else default_reward_function
        
        # Load World Model
        self.world_model = WorldModel()
        try:
            self.world_model.load_state_dict(torch.load(model_path))
            self.world_model.eval() # Set to evaluation mode
            print(f"Loaded World Model from {model_path}")
        except FileNotFoundError:
            print(f"WARNING: {model_path} not found. Using random dynamics!")
            self.world_model = None

        # --- ACTION SPACE ---
        # 4 Discrete actions, each with 5 levels (0, 1, 2, 3, 4)
        # Order: Stirring, Solution, Flow, Light
        self.action_space = spaces.MultiDiscrete([5, 5, 5, 5])

        # --- OBSERVATION SPACE (STATE) ---
        # Single continuous variable: Density
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal state variables
        self.current_density = None
        self.step_count = 0
        self.max_steps_per_episode = 720 

    def _get_obs(self):
        """Returns the normalized observation."""
        obs = np.array([self.current_density], dtype=np.float32)
        cleaned = self.data_handler.clean_observation(obs)
        return self.data_handler.normalize_observation(cleaned)

    def _get_info(self):
        """Returns auxiliary info."""
        return {"density": self.current_density}

    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        """
        super().reset(seed=seed)
        
        # Initial state: Low density inoculation (random variance)
        # We start small, similar to start of episodes in CSV
        self.current_density = 0.01 + np.random.uniform(0, 0.005)
        
        self.step_count = 0
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Simulates one time step with Discrete Actions using World Model.
        Action is array: [Stirring, Solution, Flow, Light] (ints 0-4)
        """
        
        # Unpack actions
        stirring, solution, flow, light = action
        
        # Calculate Next State using World Model
        if self.world_model:
            # Inputs to model were scaled: Actions / 4.0
            # Density was raw (but let's assume raw is ~0-1)
            
            with torch.no_grad():
                input_tensor = torch.tensor([
                    self.current_density,
                    stirring / 4.0, 
                    solution / 4.0, 
                    flow / 4.0, 
                    light / 4.0
                ], dtype=torch.float32).unsqueeze(0) # Batch dim
                
                prediction = self.world_model(input_tensor)
                next_density = prediction.item()
                
            old_density = self.current_density
            self.current_density = next_density
            
        else:
            # Fallback (Should not happen if setup is correct)
            old_density = self.current_density
            self.current_density += 0.0 # No op
        
        # Physical constraints (0 to 1)
        self.current_density = max(0.0, min(1.0, self.current_density))
        
        # --- REWARD FUNCTION ---
        reward = self.reward_func(self.current_density, old_density, self.step_count, self.max_steps_per_episode)
        
        # Termination conditions
        self.step_count += 1
        terminated = False
        if self.current_density <= 0.0:
            terminated = True
            # Crash penalty is already handled in default_reward_function, but checking if user overrides it?
            # actually usually better to let reward function handle it entirely.
            # in my default_reward_function I put -10 there.
            # but WAIT, looking at original code:
            # terminated = True
            # reward = -10.0
            # My extracted function does this check too.
            # So I should just trust the reward function return.
            pass

        truncated = False
        if self.step_count >= self.max_steps_per_episode:
            truncated = True
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.step_count}, Density: {self.current_density:.4f}")

    def close(self):
        pass
