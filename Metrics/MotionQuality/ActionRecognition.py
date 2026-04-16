import os
import sys
CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(os.path.join(CURRENT_DIR, "tools", "mmaction2"))
sys.path.append(os.path.join(BASE_DIR, "Tools"))
sys.path.append(os.path.join(BASE_DIR, "Metrics", "StoryVideoAlignment", "tools", "Grounded-SAM-2"))
import torch
import numpy as np
import tempfile
import shutil
from operator import itemgetter
from typing import Optional, List, Dict, Tuple

from mmengine import Config
from mmaction.apis import inference_recognizer, init_recognizer
from transformers import CLIPModel, AutoTokenizer
from gemini_api import GeminiAPI
from grounded_sam2_tracking import GroundedSAM2Tracker


class ActionRecognition:
    """Action recognition scorer for single video evaluation with character-specific analysis."""
    
    def __init__(self, config_path=None, checkpoint_path=None, device=None, gemini_api_keys=None, gemini_proxy=None):
        """
        Initialize the action recognition model, CLIP model, Gemini API, and SAM2 tracker.
        
        Args:
            config_path (str): Path to action model config file
            checkpoint_path (str): Path to action model checkpoint
            device (str): Device to run models on ('cuda' or 'cpu')
            gemini_api_keys (str or list): Gemini API key(s)
            gemini_proxy (str): Proxy for Gemini API
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default paths
        if config_path is None:
            config_path = os.path.join(CURRENT_DIR, 'tools', 'mmaction2', 'configs', 'recognition', 'videomaev2', 'vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py')
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CURRENT_DIR, 'checkpoints', 'VideoMAE', 'vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth')
        
        # Load action recognition model
        cfg = Config.fromfile(config_path)
        self.action_model = init_recognizer(cfg, checkpoint_path, device=self.device)
        
        # Load CLIP model for text similarity
        self.clip_model = CLIPModel.from_pretrained(os.path.join(CURRENT_DIR, "checkpoints", "clip-vit-base-patch32")).to(self.device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(os.path.join(CURRENT_DIR, "checkpoints", "clip-vit-base-patch32"))
        
        # Load class labels
        label_path = os.path.join(CURRENT_DIR, 'tools', 'mmaction2', 'tools', 'data', 'kinetics', 'label_map_k400.txt')
        with open(label_path) as f:
            self.labels = [x.strip() for x in f.readlines()]
        
        # Initialize Gemini API
        if gemini_api_keys is None:
            gemini_api_keys = ["YOUR_GEMINI_API_KEY"]
        self.gemini = GeminiAPI(api_keys=gemini_api_keys, proxy=gemini_proxy)
        
        # Initialize SAM2 tracker
        self.tracker = GroundedSAM2Tracker(
            model_id="IDEA-Research/grounding-dino-tiny",
            prompt_type="box"
        )
    
    def analyze_scene_with_gemini(self, video_path: str, prompt: str) -> Dict:
        """
        Analyze video scene using Gemini API to determine if character tracking is needed.
        
        Args:
            video_path (str): Path to video file
            prompt (str): Scene description prompt
            
        Returns:
            dict: Analysis results containing need_tracking, characters, and target_actions
        """
        analysis_prompt = f"""
        Analyze this video and the following scene description:
        
        Scene Description: {prompt}
        
        Please determine:
        1. How many distinct MAIN CHARACTERS (protagonists) are performing clear, identifiable actions in this video?
        2. Are these main characters performing independent actions (not interacting directly with each other or shared objects)?
        3. If there are multiple independent main characters, what are their individual target actions?
        
        Respond in the following JSON format:
        {{
            "character_count": <number>,
            "need_individual_tracking": <true/false>,
            "characters": [
                {{
                    "description": "<character description for detection>",
                    "target_action": "<specific action this character is performing>"
                }}
            ],
            "overall_target_action": "<overall scene action if not using individual tracking>"
        }}
        
        Guidelines:
        - ONLY include MAIN CHARACTERS (protagonists) - typically 1-2 characters who are the focus of the scene
        - EXCLUDE background characters, extras, or people without clear/significant actions
        - AVOID duplicates - each character should be unique and distinct
        - Set need_individual_tracking to true only if there are more than one main characters AND they are performing independent actions
        - Character descriptions should be simple for object detection (e.g., "person", "woman", "child", "man")
        - Target actions should be specific action verbs (e.g., "eating", "walking", "sitting", "reading")
        - If a character only appears in background or has no clear action, DO NOT include them
        """
        
        try:
            response = self.gemini.generate_from_videos([video_path], analysis_prompt)
            # Parse JSON response
            import json
            
            # Extract JSON from response if it contains other text
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                analysis = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
            
            return analysis
        except Exception as e:
            print(f"Error in Gemini analysis: {e}")
            # Fallback to simple analysis
            return {
                "character_count": 1,
                "need_individual_tracking": False,
                "characters": [],
                "overall_target_action": "general action"
            }
    
    def track_characters(self, video_path: str, characters: List[Dict]) -> List[str]:
        """
        Track individual characters using Grounded SAM2 and return cropped video paths.
        
        Args:
            video_path (str): Path to original video
            characters (list): List of character descriptions
            
        Returns:
            list: Paths to cropped videos for each character
        """
        cropped_videos = []
        
        for i, character in enumerate(characters):
            # Create temporary directory for this character
            temp_dir = tempfile.mkdtemp(prefix=f"character_{i}_")
            
            # Set up paths
            frames_dir = os.path.join(temp_dir, "frames")
            tracking_dir = os.path.join(temp_dir, "tracking")
            cropped_dir = os.path.join(temp_dir, "cropped")
            cropped_video_path = os.path.join(temp_dir, f"character_{i}_cropped.mp4")
            
            # Process character tracking
            self.tracker.process_video(
                video_path=video_path,
                text_prompt=character["description"] + '.',
                source_frames_dir=frames_dir,
                tracking_results_dir=tracking_dir,
                cropped_frames_dir=cropped_dir,
                output_video_path=os.path.join(temp_dir, "annotated.mp4"),
                cropped_video_path=cropped_video_path
            )
            
            if os.path.exists(cropped_video_path):
                cropped_videos.append(cropped_video_path)
            else:
                print(f"Warning: Failed to create cropped video for character {i}")
                cropped_videos.append(None)
        
        return cropped_videos
    
    def predict_actions(self, video_path, top_k=5):
        """
        Predict top-k actions from video.
        
        Args:
            video_path (str): Path to video file
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of tuples (action_label, confidence_score)
        """
        pred_result = inference_recognizer(self.action_model, video_path)
        pred_scores = pred_result.pred_scores.item.tolist()
        
        # Get top-k predictions
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        top_k_indices = score_sorted[:top_k]
        
        results = [(self.labels[idx], score) for idx, score in top_k_indices]
        return results
    
    def calculate_action_score(self, video_path, target_action):
        """
        Calculate action recognition score for a video against target action.
        
        Args:
            video_path (str): Path to video file
            target_action (str): Target action description
            
        Returns:
            float: Action recognition score
        """
        # Get top-5 action predictions
        predictions = self.predict_actions(video_path, top_k=5)
        
        # Extract action labels and confidence scores
        action_labels = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]
        
        # Tokenize predictions and target action
        action_tokens = self.clip_tokenizer(action_labels, return_tensors="pt", padding=True, truncation=True)
        target_tokens = self.clip_tokenizer(target_action, return_tensors="pt", padding=True, truncation=True)
        
        action_input = action_tokens["input_ids"].to(self.device)
        target_input = target_tokens["input_ids"].to(self.device)
        
        # Calculate text features
        with torch.no_grad():
            action_features = self.clip_model.get_text_features(action_input)
            target_features = self.clip_model.get_text_features(target_input)
        
        # Normalize features
        action_features = action_features / action_features.norm(p=2, dim=-1, keepdim=True)
        target_features = target_features / target_features.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate similarity scores
        similarities = action_features @ target_features.T
        
        # Weight similarities by confidence scores
        confidence_tensor = torch.tensor(confidences).unsqueeze(0).float().to(self.device)
        weighted_score = confidence_tensor @ similarities
        
        return weighted_score[0][0].item()
    
    def evaluate_video(self, video_path, target_action=None, return_predictions=False):
        """
        Evaluate a single video for action recognition.
        
        Args:
            video_path (str): Path to video file
            target_action (str, optional): Target action for scoring
            return_predictions (bool): Whether to return top predictions
            
        Returns:
            dict: Evaluation results containing score and optionally predictions
        """
        results = {}
        
        # Get action predictions
        predictions = self.predict_actions(video_path)
        print("Top-5 Action Predictions:", predictions)
        
        if return_predictions:
            results['predictions'] = predictions
        
        # Calculate score if target action provided
        if target_action:
            score = self.calculate_action_score(video_path, target_action)
            results['action_score'] = score
        
        return results
    
    def evaluate_video_with_prompt(self, video_path: str, prompt: str, return_predictions: bool = False) -> Dict:
        """
        Evaluate video with scene prompt, using character-specific analysis when appropriate.
        
        Args:
            video_path (str): Path to video file
            prompt (str): Scene description prompt
            return_predictions (bool): Whether to return detailed predictions
            
        Returns:
            dict: Comprehensive evaluation results
        """
        results = {
            "video_path": video_path,
            "prompt": prompt,
            "analysis_method": "unknown",
            "scores": {},
            "predictions": {} if return_predictions else None
        }
        
        # Step 1: Analyze scene with Gemini
        print("Analyzing scene with Gemini API...")
        scene_analysis = self.analyze_scene_with_gemini(video_path, prompt)
        print("Scene analysis results:", scene_analysis)
        results["scene_analysis"] = scene_analysis
        
        # Step 2: Decide on analysis approach
        if scene_analysis.get("need_individual_tracking", False) and len(scene_analysis.get("characters", [])) > 1:
            print("Using character-specific tracking and analysis...")
            results["analysis_method"] = "character_specific"
            
            # Track individual characters
            characters = scene_analysis["characters"]
            cropped_videos = self.track_characters(video_path, characters)
            
            # Evaluate each character individually
            character_results = []
            for i, (character, cropped_video) in enumerate(zip(characters, cropped_videos)):
                if cropped_video and os.path.exists(cropped_video):
                    char_result = self.evaluate_video(
                        video_path=cropped_video,
                        target_action=character["target_action"],
                        return_predictions=return_predictions
                    )
                    char_result["character_description"] = character["description"]
                    char_result["target_action"] = character["target_action"]
                    character_results.append(char_result)
                    
                    # Clean up temporary files
                    try:
                        shutil.rmtree(os.path.dirname(cropped_video))
                    except:
                        pass
                else:
                    print(f"Warning: Could not process character {i}")
                    character_results.append({
                        "character_description": character["description"],
                        "target_action": character["target_action"],
                        "action_score": 0.0,
                        "error": "Tracking failed"
                    })
            
            results["character_results"] = character_results
            
            # Calculate overall score as average of character scores
            valid_scores = [r.get("action_score", 0.0) for r in character_results if "action_score" in r]
            results["scores"]["overall_action_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            
        else:
            print("Using overall video analysis...")
            results["analysis_method"] = "overall"
            
            # Use overall target action
            target_action = scene_analysis.get("overall_target_action", "general action")
            
            # Evaluate entire video
            overall_result = self.evaluate_video(
                video_path=video_path,
                target_action=target_action,
                return_predictions=return_predictions
            )
            
            results["scores"] = overall_result
            if return_predictions and "predictions" in overall_result:
                results["predictions"]["overall"] = overall_result["predictions"]
        
        return results


if __name__ == '__main__':
    # Example usage
    video_path = "your_video_path_here"
    prompt = "your_prompt_here"
    
    # Initialize action recognition
    recognizer = ActionRecognition(
        gemini_api_keys=["YOUR_GEMINI_API_KEY"],
        gemini_proxy="YOUR_PROXY_URL"
    )
    
    # Evaluate video with prompt
    results = recognizer.evaluate_video_with_prompt(
        video_path=video_path,
        prompt=prompt,
        return_predictions=True
    )
    
    # Print results
    print(f"\nVideo: {video_path}")
    print(f"Prompt: {prompt}")
    print(f"Analysis Method: {results['analysis_method']}")
    print(f"Scene Analysis: {results['scene_analysis']}")
    
    if results['analysis_method'] == 'character_specific':
        print("\nCharacter-specific Results:")
        for i, char_result in enumerate(results['character_results']):
            print(f"  Character {i+1}: {char_result['character_description']}")
            print(f"    Target Action: {char_result['target_action']}")
            print(f"    Action Score: {char_result.get('action_score', 'N/A'):.4f}")
        print(f"\nOverall Action Score: {results['scores']['overall_action_score']:.4f}")
    else:
        print("\nOverall Results:")
        if 'predictions' in results and results['predictions']:
            print("Top-5 Action Predictions:")
            for i, (action, confidence) in enumerate(results['predictions']['overall']):
                print(f"  {i+1}. {action}: {confidence:.4f}")
        print(f"Action Score: {results['scores'].get('action_score', 'N/A'):.4f}")
