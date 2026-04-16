import torch
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ
from PIL import Image
import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
MUSIQ_MODEL_PATH = os.path.join(BASE_DIR, "VisualQuality", "checkpoints", "MusIQ", "musiq.pth")


class MusiqEvaluator:
    def __init__(self, model_path=None, device=None):
        """
        Initialize MUSIQ evaluator
        
        Args:
            model_path: Path to the MUSIQ model weights
            device: Device to run inference on (default: auto-detect)
        """
        self.model_path = model_path or MUSIQ_MODEL_PATH
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MUSIQ model not found at {self.model_path}")
        
        # Load and initialize model
        self.model = MUSIQ(pretrained_model_path=self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def get_timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def load_video(self, video_path):
        """Load video frames from directory or video file"""
        if os.path.isdir(video_path):
            # Load frames from directory
            frames = []
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for frame_file in frame_files:
                img_path = os.path.join(video_path, frame_file)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transforms.ToTensor()(img)
                frames.append(img_tensor)
            if not frames:
                raise ValueError(f"No image frames found in {video_path}")
            return torch.stack(frames)
        else:
            # For actual video files
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to tensor [0,1]
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(frame_tensor)
                cap.release()
                if not frames:
                    raise ValueError(f"Could not load frames from video {video_path}")
                return torch.stack(frames)
            except ImportError:
                raise ImportError("OpenCV (cv2) is required to load video files. Install with: pip install opencv-python")

    def transform(self, images, preprocess_mode='longer'):
        """Transform images for MUSIQ model"""
        if preprocess_mode.startswith('shorter'):
            _, _, h, w = images.size()
            if min(h,w) > 512:
                scale = 512./min(h,w)
                images = transforms.Resize(size=( int(scale * h), int(scale * w) ), antialias=False)(images)
                if preprocess_mode == 'shorter_centercrop':
                    images = transforms.CenterCrop(512)(images)

        elif preprocess_mode == 'longer':
            _, _, h, w = images.size()
            if max(h,w) > 512:
                scale = 512./max(h,w)
                images = transforms.Resize(size=( int(scale * h), int(scale * w) ), antialias=False)(images)

        elif preprocess_mode == 'None':
            return images / 255.

        else:
            raise ValueError("Please recheck imaging_quality_mode")
        return images / 255.

    def evaluate_video(self, video_path, preprocess_mode='longer', verbose=True):
        """
        Evaluate a single video and return quality score
        
        Args:
            video_path: Path to video file or directory containing frames
            preprocess_mode: Image preprocessing mode ('longer', 'shorter', 'shorter_centercrop', 'None')
            verbose: Whether to print progress information
            
        Returns:
            float: Normalized quality score (0-1 range)
        """
        try:
            # Load video frames
            images = self.load_video(video_path)
            images = self.transform(images, preprocess_mode)
            
            # Handle videos with no frames
            if len(images) == 0:
                raise ValueError(f"No frames found in {video_path}")
                
            # Evaluate each frame
            acc_score_video = 0.
            for i in range(len(images)):
                frame = images[i].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    score = self.model(frame)
                acc_score_video += float(score)
            
            # Calculate average score for the video
            video_score = acc_score_video / len(images)
            normalized_score = video_score / 100.0
            
            if verbose:
                video_name = os.path.basename(video_path)
                timestamp = self.get_timestamp()
                print(f"{timestamp} Video: {video_name}, Quality score: {normalized_score:.4f}")
            
            return normalized_score
            
        except Exception as e:
            if verbose:
                print(f"Error processing {video_path}: {e}")
            raise e

    def evaluate_multiple_videos(self, video_list, preprocess_mode='longer', output_file=None, verbose=True):
        """
        Evaluate multiple videos and return average score
        
        Args:
            video_list: List of video paths
            preprocess_mode: Image preprocessing mode
            output_file: Optional file to save results
            verbose: Whether to print progress information
            
        Returns:
            tuple: (average_score, list_of_individual_results)
        """
        video_results = []
        total_score = 0.0
        processed_count = 0
        
        for video_path in tqdm(video_list, disable=not verbose):
            try:
                score = self.evaluate_video(video_path, preprocess_mode, verbose=False)
                
                total_score += score
                processed_count += 1
                current_avg = total_score / processed_count
                
                if verbose:
                    video_name = os.path.basename(video_path)
                    timestamp = self.get_timestamp()
                    log_line = f"{timestamp} Vid: {video_name}, Score: {score:.4f}, Avg: {current_avg:.4f}"
                    print(log_line)
                    
                    if output_file:
                        with open(output_file, 'a') as f:
                            f.write(log_line + "\n")
                            
                video_results.append({'video_path': video_path, 'score': score})
                
            except Exception as e:
                if verbose:
                    print(f"Error processing {video_path}: {e}")
                continue
                
        if not video_results:
            raise ValueError("No videos were successfully processed")
            
        average_score = total_score / processed_count
        
        if verbose:
            timestamp = self.get_timestamp()
            final_line = f"{timestamp} Final average score: {average_score:.4f}, Total videos: {len(video_results)}"
            print(final_line)
            
            if output_file:
                with open(output_file, 'a') as f:
                    f.write(final_line + "\n")
        
        return average_score, video_results


if __name__ == "__main__":
    evaluator = MusiqEvaluator()
    video_path = "path/to/your/video.mp4"  # Replace with an actual video path.
    try:
        score = evaluator.evaluate_video(video_path, verbose=True)
        print(f"Final quality score: {score:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
