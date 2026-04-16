import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
T2V_METRICS_DIR = os.path.join(BASE_DIR, "StoryVideoAlignment", "tools", "t2v_metrics")
T2V_CACHE_DIR = os.path.join(T2V_METRICS_DIR, "weights", "clip-flant5-xxl")
TMP_FRAME_DIR = os.path.join(BASE_DIR, "Evaluation", "results", "tmp", "story_video_alignment")

sys.path.append(T2V_METRICS_DIR)
import cv2
import numpy as np
import t2v_metrics
from pathlib import Path

class VideoVQAScore:
    def __init__(self, model='clip-flant5-xxl'):
        """Initialize the VideoVQAScore calculator with specified model."""
        self.clip_flant5_score = t2v_metrics.VQAScore(model=model, cache_dir=T2V_CACHE_DIR)
    
    def extract_frames_from_video(self, video_path, interval_seconds=0.5):
        """Extract frames from video at specified interval (in seconds)."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            cap.release()
            return []
        
        frame_interval = int(fps * interval_seconds)
        frames = []
        
        frame_indices = range(0, total_frames, frame_interval)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def calculate_single_video_score(self, video_path, prompt, interval_seconds=0.5):
        """Calculate VQAScore for a single video against a prompt using multiple frames."""
        frames = self.extract_frames_from_video(video_path, interval_seconds)
        
        if not frames:
            return None
        
        scores = []
        temp_frame_paths = []
        
        try:
            # Create temporary directory if it doesn't exist
            temp_dir = TMP_FRAME_DIR
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save all frames temporarily and calculate scores
            for i, frame in enumerate(frames):
                temp_frame_path = f"{temp_dir}/temp_frame_{Path(video_path).stem}_{i}.png"
                cv2.imwrite(temp_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                temp_frame_paths.append(temp_frame_path)
            
            # Calculate VQAScore for all frames at once
            score = self.clip_flant5_score(images=temp_frame_paths, texts=[prompt])
            
            # Convert scores to float and calculate average
            frame_scores = [float(s) for s in score]
            avg_score = np.mean(frame_scores)
            
            # Clean up temp files
            for temp_path in temp_frame_paths:
                os.remove(temp_path)
            
            print(f"  Extracted {len(frames)} frames, scores: {[f'{s:.4f}' for s in frame_scores]}")
            return avg_score
            
        except Exception as e:
            # Clean up temp files in case of error
            for temp_path in temp_frame_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            raise e
    
    def calculate_directory_scores(self, video_dir, prompt, video_extensions=None):
        """Calculate VQAScore for all videos in a directory against a prompt."""
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        # Get all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
        
        results = []
        print(f"Found {len(video_files)} video files in {video_dir}")
        
        for video_path in sorted(video_files):
            print(f"\nProcessing: {video_path.name}")
            
            try:
                score_value = self.calculate_single_video_score(str(video_path), prompt)
                if score_value is not None:
                    results.append((video_path.name, score_value))
                    print(f"VQAScore: {score_value:.4f}")
                else:
                    print(f"Failed to extract frame from {video_path.name}")
            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
        
        return results
    
    def print_results_summary(self, results):
        """Print a formatted summary of results."""
        print(f"\n{'='*50}")
        print("SUMMARY RESULTS:")
        print(f"{'='*50}")
        for video_name, score in sorted(results, key=lambda x: x[1], reverse=True):
            print(f"{video_name}: {score:.4f}")

# Usage example
if __name__ == "__main__":
    # Initialize the VQAScore calculator
    vqa_calculator = VideoVQAScore()
    
    # Single video file path
    video_path = "path/to/your/video.mp4"  # Replace with an actual video path.
    prompt = "your prompt here"  # Replace with an actual prompt.
    
    # Calculate VQAScore for single video
    print(f"Processing video: {video_path}")
    try:
        score = vqa_calculator.calculate_single_video_score(video_path, prompt)
        if score is not None:
            print(f"\nFinal VQAScore: {score:.4f}")
        else:
            print("Failed to calculate score for the video")
    except Exception as e:
        print(f"Error processing video: {e}")