import torch.nn as nn

class WorldModel(nn.Module):
    """
    Neural Network identifying the environment dynamics (Transition Function).
    Same architecture as in train_world_model.py
    """
    def __init__(self):
        super(WorldModel, self).__init__()
        # Input: 5 (Density + 4 Actions)
        # Output: 1 (Next Density)
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        return self.net(x)
