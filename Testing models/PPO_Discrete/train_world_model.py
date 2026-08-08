import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Configuration
CSV_PATH = r"e:\SEGP\PBR_Simulation_Dataset\PBR_Simulation_Dataset\pbr_simulation_dataset.csv"
MODEL_PATH = "world_model.pth"
BATCH_SIZE = 64
EPOCHS = 100 # Quick training
LEARNING_RATE = 0.001

class PBRDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # We need to predict Next_Density from Current_Density + Actions
        # Group by episode to ensure we don't mix boundaries
        groups = df.groupby('episode_id')
        
        inputs = []
        targets = []
        
        print("Processing episodes...")
        for _, group in groups:
            # Sort by timestep just in case
            group = group.sort_values('timestep')
            
            # Extract features
            # State: Density
            density = group['state_total_microalgae_density'].values
            
            # Actions: Stirring, Solution, Flow, Light (0-4 integers)
            # We can use them directly or one-hot. For simplicity, we keep them as normalized floats 0-1 or just scaled.
            # Let's simple-scale them (divide by 4.0)
            a_stir = group['action_stirring_rate'].values / 4.0
            a_sol = group['action_solution_amount'].values / 4.0
            a_flow = group['action_rate_of_flow'].values / 4.0
            a_light = group['action_light_intensity'].values / 4.0
            
            # Timesteps T. We can use T to predict T+1
            # Input at t: [Density_t, Stir_t, Sol_t, Flow_t, Light_t]
            # Target at t: [Density_t+1]
            
            # We slice [:-1] for inputs and [1:] for targets
            curr_density = density[:-1]
            curr_a_stir = a_stir[:-1]
            curr_a_sol = a_sol[:-1]
            curr_a_flow = a_flow[:-1]
            curr_a_light = a_light[:-1]
            
            next_density = density[1:]
            
            # Stack features
            # Shape: (N, 5) -> [Density, Stir, Sol, Flow, Light]
            episode_inputs = np.column_stack([
                curr_density, 
                curr_a_stir, 
                curr_a_sol, 
                curr_a_flow, 
                curr_a_light
            ])
            
            episode_targets = next_density.reshape(-1, 1)
            
            inputs.append(episode_inputs)
            targets.append(episode_targets)
            
        self.inputs = np.vstack(inputs).astype(np.float32)
        self.targets = np.vstack(targets).astype(np.float32)
        
        print(f"Data Loaded. Samples: {len(self.inputs)}")
        print(f"Input Shape: {self.inputs.shape}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

from world_model import WorldModel

def train():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    dataset = PBRDataset(CSV_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = WorldModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
            
    print("Training complete.")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
