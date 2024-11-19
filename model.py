import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import glob

class SnakeUNet(nn.Module):
    def __init__(self, n_channels=7):  # 4 channels for key input + 3 for RGB image
        super(SnakeUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),  # 32 because of skip connection
            nn.ReLU(),
            nn.Conv2d(3, 3, 1),  # Final 3 channels for RGB output
            nn.Sigmoid()  # Ensure output values between 0-1
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # Decoder with skip connections
        d1 = self.dec1(e2)
        d2 = self.dec2(torch.cat([d1, e1], dim=1))
        
        return d2

class SnakeFramePredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SnakeUNet().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def load_session_data(self, session_dir):
        print(f"\nLoading session data from: {session_dir}")
        frames = []
        keys = []
        
        # Load frames
        frame_files = sorted(glob.glob(os.path.join(session_dir, "frames", "*.jpg")))
        print(f"Found {len(frame_files)} frame files")
        
        for frame_file in frame_files:
            img = Image.open(frame_file)
            # print(f"Loading {frame_file} - Original mode: {img.mode}")
            # Convert to RGBA if not already
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
                # print(f"Converted to RGBA mode")
            img = np.array(img) / 255.0  # Normalize to 0-1
            # print(f"Frame shape after conversion: {img.shape}")
            frames.append(img)
            
        # Load key inputs
        key_data = {}
        keylog_path = os.path.join(session_dir, "keylog.txt")
        # print(f"\nLoading keylog from: {keylog_path}")
        
        with open(keylog_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                frame_num, key = line.strip().split(',')
                key_data[int(frame_num)] = key
        # print(f"Loaded {len(key_data)} key entries")
        
        # Convert keys to one-hot encoding
        key_mapping = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'none': 4}
        # print("\nConverting keys to one-hot encoding")
        for i in range(len(frames)):
            key = key_data.get(i, 'none')
            key_vector = np.zeros(4)  # Only 4 channels needed for directional keys
            if key in key_mapping and key != 'none':
                key_vector[key_mapping[key]] = 1
            keys.append(key_vector)
            # if i < 5:  # Print first few for verification
                # print(f"Frame {i}: Key '{key}' -> Vector {key_vector}")
        
        frames_array = np.array(frames[:-1])
        keys_array = np.array(keys[:-1])
        next_frames_array = np.array(frames[1:])
        
        # print(f"\nFinal shapes:")
        # print(f"Current frames: {frames_array.shape}")
        # print(f"Keys: {keys_array.shape}")
        # print(f"Next frames: {next_frames_array.shape}")
        
        return frames_array, keys_array, next_frames_array

    def prepare_input(self, frame, key_vector):
        # Ensure frame is RGB
        if frame.shape[-1] != 3:
            print("Converting to RGB")
            frame = frame[:, :, :3]
            print(f"New frame shape: {frame.shape}")
        
        # Combine frame and key input into single input tensor
        frame_tensor = torch.FloatTensor(frame).permute(2, 0, 1)  # RGB channels
        key_tensor = torch.FloatTensor(key_vector)
        key_channels = key_tensor.view(-1, 1, 1).repeat(1, frame.shape[0], frame.shape[1])
        x = torch.cat([frame_tensor, key_channels], dim=0)  # Should be 7 channels total
        
        return x.unsqueeze(0).to(self.device)

    def train(self, session_dirs, epochs=20, batch_size=32, checkpoint_dir='checkpoints', 
          patience=5, min_delta=0.001):
        print(f"\nStarting training with {len(session_dirs)} sessions")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Using device: {self.device}")
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            total_loss = 0
            batch_count = 0
            
            for session_dir in session_dirs:
                print(f"\nProcessing session: {session_dir}")
                current_frames, keys, next_frames = self.load_session_data(session_dir)
                
                # Training loop
                for i in range(0, len(current_frames), batch_size):
                    batch_count += 1
                    # print(f"\nProcessing batch {batch_count}")
                    
                    batch_frames = current_frames[i:i+batch_size]
                    batch_keys = keys[i:i+batch_size]
                    batch_next = next_frames[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    
                    # Prepare batch input
                    x = torch.cat([
                        self.prepare_input(f, k) 
                        for f, k in zip(batch_frames, batch_keys)
                    ])
                    
                    y = torch.FloatTensor(batch_next).permute(0, 3, 1, 2).to(self.device)
                    
                    # Forward pass
                    pred = self.model(x)
                    loss = F.mse_loss(pred, y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    print(f"Batch loss: {loss.item():.6f}")
                    
                    # Save checkpoint every 500 batches
                    if batch_count % 500 == 0:
                        checkpoint_path = os.path.join(
                            checkpoint_dir, 
                            f'checkpoint_e{epoch+1}_b{batch_count}.pth'
                        )
                        torch.save({
                            'epoch': epoch + 1,
                            'batch': batch_count,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                        }, checkpoint_path)
                        print(f"Saved checkpoint to {checkpoint_path}")
            
            # Calculate epoch loss after processing all sessions
            epoch_loss = total_loss / batch_count
            print(f"\nEpoch {epoch+1} - Average loss: {epoch_loss:.6f}")
            
            # Early stopping check
            if epoch_loss < best_loss - min_delta:  # Improved by at least min_delta
                print(f"Loss improved from {best_loss:.6f} to {epoch_loss:.6f}")
                best_loss = epoch_loss
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, best_model_path)
                print(f"New best model saved to {best_model_path}")
            else:
                patience_counter += 1
                print(f"Loss did not improve. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered! No improvement for {patience} epochs")
                    print(f"Best loss achieved: {best_loss:.6f}")
                    # Load best model before stopping
                    self.load_checkpoint(os.path.join(checkpoint_dir, 'best_model.pth'))
                    return
            
            # Save epoch checkpoint
            epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, epoch_path)
            print(f"Saved epoch checkpoint to {epoch_path}")

    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'batch' in checkpoint:
            print(f"Batch: {checkpoint['batch']}")
        print(f"Loss: {checkpoint['loss']:.6f}")
        
        return checkpoint

    def predict_next_frame(self, current_frame, key_press):
        print(f"\nMaking prediction:")
        print(f"Input frame shape: {current_frame.shape}")
        print(f"Input key press: {key_press}")
        
        with torch.no_grad():
            x = self.prepare_input(current_frame, key_press)
            print(f"Model input shape: {x.shape}")
            
            pred = self.model(x)
            print(f"Prediction shape: {pred.shape}")
            
            result = pred[0].permute(1, 2, 0).cpu().numpy()
            print(f"Final output shape: {result.shape}")
            
            return result
