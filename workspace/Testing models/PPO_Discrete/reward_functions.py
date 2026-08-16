def default_reward_function(current_density, old_density, step_count, max_steps):
    """
    Calculates the reward based on the change in density.
    
    Args:
        current_density (float): The current microalgae density.
        old_density (float): The previous microalgae density.
        step_count (int): Current step in the episode.
        max_steps (int): Maximum steps allowed in the episode.
        
    Returns:
        float: The calculated reward.
    """
    # Reward is change in density (growth) * scale factor
    reward = (current_density - old_density) * 100.0
    
    if current_density <= 0.0:
        reward = -10.0 # Crash penalty
        
    return reward
