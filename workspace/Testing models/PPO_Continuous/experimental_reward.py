def calculated_reward_function(productivity, ph, light_in, nutrients, od):
    """
    Calculates the composite reward for the Photobioreactor.
    
    Components:
    1. Base: Biomass Productivity (g/L/h) * 100
    2. Penalty: pH Stress (-10.0 if not [6.8, 8.2])
    3. Penalty: Photoinhibition/Energy Cost (-2.0 if Light > 600)
    4. Penalty: Nutrient Starvation (-20.0 if Nutrients < 10.0)
    5. Penalty: Culture Crash (-50.0 if OD < 0.05)
    
    Args:
        productivity (float): Biomass productivity in g/L per hour.
        ph (float): Current pH level.
        light_in (float): Input light intensity (umol/m2/s).
        nutrients (float): Current nutrient concentration (mg/L).
        od (float): Current optical density / turbidity.
        
    Returns:
        float: Total reward value.
    """
    
    # Base Reward
    reward = productivity * 100.0
    
    # Health checks (Penalties)
    
    # pH sensitivity (Stricter range)
    if not (6.8 <= ph <= 8.2):
        reward -= 10.0
        
    # Energy efficiency / Photoinhibition penalty
    if light_in > 600.0:
        reward -= 2.0
        
    # Nutrient starvation penalty (Stricter threshold)
    if nutrients < 10.0:
        reward -= 20.0
        
    # Culture crash penalty
    if od < 0.05:
        reward -= 50.0
        
    return reward
