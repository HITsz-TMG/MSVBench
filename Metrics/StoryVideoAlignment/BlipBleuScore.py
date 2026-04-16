import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import cv2
import numpy as np
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from pycocoevalcap.bleu.bleu import Bleu
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
BLIP2_MODEL_PATH = os.path.join(BASE_DIR, "StoryVideoAlignment", "checkpoints", "blip2-opt-2.7b")

class BlipBleuScore:
    def __init__(self, model_path=None):
        """Initialize the BlipBleuScore calculator with BLIP2 model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip2_processor = AutoProcessor.from_pretrained(model_path or BLIP2_MODEL_PATH)
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            model_path or BLIP2_MODEL_PATH, torch_dtype=torch.float16
        ).to(self.device)
        
        # Initialize BLEU scorers
        self.bleu1 = Bleu(n=1)
        self.bleu2 = Bleu(n=2)
        self.bleu3 = Bleu(n=3)
        self.bleu4 = Bleu(n=4)
    
    def extract_frames_from_video(self, video_path, num_frames=5):
        """Extract specified number of frames evenly distributed from video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return []
        
        # Calculate frame indices to extract evenly distributed frames
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame and convert BGR to RGB
                resized_frame = cv2.resize(frame, (224, 224))
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                frames.append(resized_frame)
        
        cap.release()
        return frames
    
    def compute_max_bleu(self, gt_prompts, pred_prompts):
        """Compute maximum BLEU score between ground truth and predicted prompts."""
        scores = []
        for pred_prompt in pred_prompts:
            for gt_prompt in gt_prompts:
                cand = {0: [pred_prompt]}
                ref = {0: [gt_prompt]}
                
                # Calculate BLEU scores - handle both single values and lists
                bleu1_score, _ = self.bleu1.compute_score(ref, cand)
                bleu2_score, _ = self.bleu2.compute_score(ref, cand)
                bleu3_score, _ = self.bleu3.compute_score(ref, cand)
                bleu4_score, _ = self.bleu4.compute_score(ref, cand)
                
                # Extract scalar values if they are lists
                if isinstance(bleu1_score, list):
                    bleu1_score = bleu1_score[0] if len(bleu1_score) > 0 else 0.0
                if isinstance(bleu2_score, list):
                    bleu2_score = bleu2_score[0] if len(bleu2_score) > 0 else 0.0
                if isinstance(bleu3_score, list):
                    bleu3_score = bleu3_score[0] if len(bleu3_score) > 0 else 0.0
                if isinstance(bleu4_score, list):
                    bleu4_score = bleu4_score[0] if len(bleu4_score) > 0 else 0.0
                
                # Average BLEU score
                avg_bleu = (bleu1_score + bleu2_score + bleu3_score + bleu4_score) / 4
                scores.append(avg_bleu)
        
        return np.max(scores)
    
    def calculate_single_video_score(self, video_path, prompt, num_frames=5):
        """Calculate BLIP-BLEU score for a single video against a prompt."""
        frames = self.extract_frames_from_video(video_path, num_frames)
        
        if not frames:
            return None
        
        # Generate captions for extracted frames
        captions = []
        for frame in frames:
            # Convert numpy array to tensor
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            
            inputs = self.blip2_processor(images=frame_tensor, return_tensors="pt").to(
                self.device, torch.float16
            )
            generated_ids = self.blip2_model.generate(**inputs)
            generated_text = self.blip2_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            captions.append(generated_text)
        
        # Calculate BLIP-BLEU score
        original_text = [prompt]
        blip_bleu_score = self.compute_max_bleu(original_text, captions)
        
        print(f"  Extracted {len(frames)} frames, generated captions:")
        for i, caption in enumerate(captions):
            print(f"    Frame {i+1}: {caption}")
        
        return blip_bleu_score
    
    def calculate_directory_scores(self, video_dir, prompt, video_extensions=None):
        """Calculate BLIP-BLEU score for all videos in a directory against a prompt."""
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
                    print(f"BLIP-BLEU Score: {score_value:.4f}")
                else:
                    print(f"Failed to extract frames from {video_path.name}")
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
    # Initialize the BLIP-BLEU calculator
    blip_bleu_calculator = BlipBleuScore()
    
    # Directory containing video clips
    video_dir = "path/to/your/video_dir"  # Replace with an actual video directory.
    prompt = "your prompt here"  # Replace with an actual prompt.
    
    # Calculate BLIP-BLEU score for all videos in directory
    results = blip_bleu_calculator.calculate_directory_scores(video_dir, prompt)
    
    # Print summary
    blip_bleu_calculator.print_results_summary(results)

