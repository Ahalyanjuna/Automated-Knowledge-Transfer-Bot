import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# --- The Neural Network (The Brain) ---
class DQN(nn.Module):
    def __init__(self, input_dim=768): # 384 (Query) + 384 (Document) = 768
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs the "Helpfulness Score"
        )

    def forward(self, x):
        return self.fc(x)

# --- The RL Agent Controller ---
class RLAgent:
    def __init__(self, state_dim=384, model_path="rl_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Input is Query Vector + Doc Vector
        self.model = DQN(input_dim=state_dim * 2).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Load existing model if it exists
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"📈 Loaded existing RL model from {self.model_path}")
        else:
            print("🆕 Created new RL model (untrained).")

    def get_q_value(self, query_vec, doc_vec):
        """Predicts a score for a Query + Document pair."""
        self.model.eval()
        with torch.no_grad():
            # Combine vectors into a single 768-dim input
            state = torch.FloatTensor(np.hstack((query_vec, doc_vec))).to(self.device)
            return self.model(state).item()

    def update(self, query_vec, doc_vec, reward):
        """Trains the model on a single piece of feedback (y/n)."""
        self.model.train()
        
        state = torch.FloatTensor(np.hstack((query_vec, doc_vec))).to(self.device)
        target = torch.FloatTensor([reward]).to(self.device)

        self.optimizer.zero_grad()
        output = self.model(state)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        
        # Save the updated model
        torch.save(self.model.state_dict(), self.model_path)
        return loss.item()

if __name__ == "__main__":
    # Test initialization
    agent = RLAgent()
    print("✅ RL Agent is ready.")