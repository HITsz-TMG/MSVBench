import sys
import os
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, 'tools', 'RAFT', 'core'))

import cv2
import numpy as np
import torch
from PIL import Image
import argparse

from raft import RAFT
from tools.RAFT.core.utils.utils import InputPadder

# Set target resolution
TARGET_WIDTH = 1280
TARGET_HEIGHT = 960


class ActionStrengthEvaluator:
    def __init__(self, model_path=os.path.join(CURRENT_DIR, 'checkpoints', 'RAFT', 'models', 'raft-things.pth')):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load RAFT model for optical flow calculation"""
        args = argparse.Namespace()
        args.small = False
        args.mixed_precision = False
        args.alternate_corr = False
        
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(model_path))
        model = model.module
        model.to(self.device)
        model.eval()
        model.args.mixed_precision = False
        
        return model
    
    def calculate_flow_score(self, video_path):
        # Load the video
        cap = cv2.VideoCapture(video_path)

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Original video resolution: {original_width}x{original_height}")
        print(f"Resizing all frames to target resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
        
        # Extract frames from the video
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) < 2:
            print("Video must have at least 2 frames")
            return 0.0
        
        optical_flows = []
        
        with torch.no_grad():
            for i in range(len(frames) - 1):
                image1 = frames[i]
                image2 = frames[i + 1]
                
                image1 = torch.tensor(image1).permute(2,0,1).float().unsqueeze(0).to(self.device)
                image2 = torch.tensor(image2).permute(2,0,1).float().unsqueeze(0).to(self.device)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                
                flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                
                # Compute the magnitude of optical flow vectors
                flow_magnitude = torch.norm(flow_up.squeeze(0), dim=0)
                # Calculate the mean optical flow value for the current pair of frames
                mean_optical_flow = flow_magnitude.mean().item()
                optical_flows.append(mean_optical_flow)
        
        # Calculate the average optical flow for the entire video
        mean_optical_flow_video = np.mean(optical_flows)
        print(f"Mean optical flow for the video: {mean_optical_flow_video}")
        
        return mean_optical_flow_video


if __name__ == '__main__':
    # Example usage
    video_path = 'your_video_path_here'
    
    # Initialize evaluator
    evaluator = ActionStrengthEvaluator()
    
    # Calculate flow score
    score = evaluator.calculate_flow_score(video_path)
    print(f"Action Strength Score (Flow Score): {score}")
