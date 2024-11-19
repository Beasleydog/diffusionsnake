import torch
import cv2
import numpy as np
from PIL import Image
from model import SnakeFramePredictor

# Initialize predictor and load trained model
predictor = SnakeFramePredictor()
predictor.load_checkpoint('checkpoints/best_model.pth')

# Load current frame using PIL
current_frame = Image.open('predicted_frame.png').convert('RGB')  # Force RGB mode
current_frame = np.array(current_frame) / 255.0  # Normalize to 0-1

# Make predictions
key_vector = [0, 0, 0, 0]  # LEFT press
next_frame = predictor.predict_next_frame(current_frame, key_vector)

# Before saving
print("Output shape:", next_frame.shape)
print("Output dtype:", next_frame.dtype)
print("Output range:", next_frame.min(), next_frame.max())

# Save RGB image
output_image = Image.fromarray((next_frame * 255).astype('uint8'), 'RGB')
output_image.save('predicted_frame.png')