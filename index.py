import torch
import glob
from model import SnakeFramePredictor

# Initialize predictor
predictor = SnakeFramePredictor()

# Train on recorded sessions
session_dirs = glob.glob("sessions/session_*")
predictor.train(session_dirs)

# Save model
torch.save(predictor.model.state_dict(), "snake_predictor.pth")

