import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gc
import logging
logging.getLogger().handlers.clear()
import time
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import re
from torchvision import datasets, transforms
from timm.models import create_model, load_checkpoint

# Add SAM and tracking paths
CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(os.path.join(BASE_DIR, "Tools", "Segment-and-Track-Anything"))
from SegTracker import SegTracker
from model_args import aot_args, sam_args, segtracker_args
from aot_tracker import _palette

# Add Gemini API path
sys.path.append(os.path.join(BASE_DIR, "Tools"))
from gemini_api import GeminiAPI

class CharacterConsistencyBenchmark:
    def __init__(self, gemini_api_keys=None, gemini_proxy=None):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(stream_handler)
        
        # Initialize Gemini API for style detection
        if gemini_api_keys:
            self.gemini_api = GeminiAPI(api_keys=gemini_api_keys, proxy=gemini_proxy)
            self.logger.info("Gemini API initialized for style detection")
        else:
            self.gemini_api = None
            self.logger.warning("No Gemini API keys provided, style detection disabled")
        
        # Configure SAM and tracking arguments
        self.sam_args = sam_args.copy()
        self.sam_args['generator_args'] = {
            'points_per_side': 30,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.9,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 200,
        }

        # Segmentation tracker arguments
        self.segtracker_args = {
            'sam_gap': 20,
            'min_area': 200,
            'max_obj_num': 255,
            'min_new_obj_iou': 0.8,
        }
        
        self.aot_args = aot_args

        # Detection parameters for SAM
        # self.box_threshold = 0.6
        # self.text_threshold = 0.5
        # self.box_size_threshold = 0.8
        self.box_threshold = 0.35
        self.text_threshold = 0.8
        self.box_size_threshold = 0.5
        self.reset_image = True
        
        # Character similarity model setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize the InceptionNext models for character similarity
        self.cartoon_model = None
        self.real_model = None
        
        # Model checkpoints
        self.cartoon_checkpoint = os.path.join(CURRENT_DIR, "tools", "Inceptionnext", "output", "cartooncharacter.pth")
        self.real_checkpoint = os.path.join(CURRENT_DIR, "tools", "Inceptionnext", "output", "realcharacter.pth")
        
        # Image transforms for character models
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.logger.info("CharacterConsistencyBenchmark initialized")

    def _load_model(self, style):
        """
        Load the appropriate model based on style
        
        Args:
            style: 'cartoon' or 'real'
            
        Returns:
            torch.nn.Module: Loaded model
        """
        if style == 'cartoon':
            if self.cartoon_model is None:
                self.cartoon_model = create_model('inception_next_tiny', pretrained=False)
                if os.path.exists(self.cartoon_checkpoint):
                    load_checkpoint(self.cartoon_model, self.cartoon_checkpoint)
                    self.logger.info(f"Loaded cartoon character model from {self.cartoon_checkpoint}")
                else:
                    self.logger.warning(f"Cartoon checkpoint not found: {self.cartoon_checkpoint}")
                self.cartoon_model = self.cartoon_model.to(self.device)
                self.cartoon_model.eval()
            return self.cartoon_model
        else:  # real
            if self.real_model is None:
                self.real_model = create_model('inception_next_tiny', pretrained=False)
                if os.path.exists(self.real_checkpoint):
                    load_checkpoint(self.real_model, self.real_checkpoint)
                    self.logger.info(f"Loaded real character model from {self.real_checkpoint}")
                else:
                    self.logger.warning(f"Real checkpoint not found: {self.real_checkpoint}")
                self.real_model = self.real_model.to(self.device)
                self.real_model.eval()
            return self.real_model

    def save_prediction(self, pred_mask, output_dir, file_name):
        save_mask = Image.fromarray(pred_mask.astype(np.uint8))
        save_mask = save_mask.convert(mode='P')
        save_mask.putpalette(_palette)
        save_mask.save(os.path.join(output_dir, file_name))
        
    def colorize_mask(self, pred_mask):
        save_mask = Image.fromarray(pred_mask.astype(np.uint8))
        save_mask = save_mask.convert(mode='P')
        save_mask.putpalette(_palette)
        save_mask = save_mask.convert(mode='RGB')
        return np.array(save_mask)
        
    def extract_characters_from_video(self, video_path, output_dir=None, sample_rate=16, character_prompt=None):
        """
        Extract characters from video frames using SAM and save them
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save character crops (if None, don't save)
            sample_rate: Process every Nth frame
            character_prompt: Specific character to look for (e.g., "Little Prince", "woman in red dress")
            
        Returns:
            list: List of character crops (PIL Images)
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Use character-specific prompt if provided, otherwise use generic person/character
        prompt = character_prompt if character_prompt else "person"
        self.logger.info(f"Extracting {prompt} from video: {video_path}")
        character_crops = []
        
        # Initialize the segmentation tracker
        segtracker = SegTracker(self.segtracker_args, self.sam_args, self.aot_args)
        segtracker.restart_tracker()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(f"Video has {frame_count} frames at {fps} FPS")
        
        # Create a dedicated directory to save character crops
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if output_dir:
            character_save_dir = os.path.join(output_dir, f"{video_name}_characters")
            os.makedirs(character_save_dir, exist_ok=True)
            self.logger.info(f"Saving characters to: {character_save_dir}")
        else:
            character_save_dir = None
            
        frame_idx = 0
        processed_count = 0
        
        with torch.cuda.amp.autocast():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect character using SAM with specific character prompt if provided
                    pred_mask, boxes = segtracker.detect_and_seg(
                        frame_rgb, 
                        prompt, 
                        self.box_threshold, 
                        self.text_threshold,
                        self.box_size_threshold, 
                        self.reset_image
                    )
                    
                    if np.max(pred_mask) > 0:  # If a character is detected
                        # Process each detected object (potential character)
                        obj_ids = np.unique(pred_mask)
                        obj_ids = obj_ids[obj_ids != 0]
                        
                        for obj_id in obj_ids:
                            # Create binary mask for this object
                            obj_mask = (pred_mask == obj_id)
                            
                            # Find bounding box for the mask
                            y_indices, x_indices = np.where(obj_mask)
                            if len(y_indices) > 0 and len(x_indices) > 0:
                                x_min, x_max = np.min(x_indices), np.max(x_indices)
                                y_min, y_max = np.min(y_indices), np.max(y_indices)
                                
                                # Check if the bounding box has valid dimensions
                                if x_max <= x_min or y_max <= y_min:
                                    continue
                                    
                                # Add some padding to the character crop
                                h, w = frame_rgb.shape[:2]
                                pad = int(min(x_max-x_min, y_max-y_min) * 0.1)
                                x_min = max(0, x_min - pad)
                                y_min = max(0, y_min - pad)
                                x_max = min(w, x_max + pad)
                                y_max = min(h, y_max + pad)
                                
                                # Double-check the dimensions
                                if x_max <= x_min or y_max <= y_min or x_max > w or y_max > h:
                                    continue
                                    
                                # Crop the character
                                try:
                                    character_crop = frame_rgb[y_min:y_max, x_min:x_max]
                                    
                                    # Ensure the crop has valid dimensions
                                    if character_crop.size == 0 or character_crop.shape[0] == 0 or character_crop.shape[1] == 0:
                                        continue
                                        
                                    # Convert to PIL Image for the transformation pipeline
                                    character_pil = Image.fromarray(character_crop)
                                    
                                    # Save character crop with unique filename
                                    if character_save_dir:
                                        character_filename = f"character_{frame_idx:05d}_obj{obj_id:02d}.jpg"
                                        character_path = os.path.join(character_save_dir, character_filename)
                                        character_pil.save(character_path)
                                        
                                    character_crops.append(character_pil)
                                except Exception as e:
                                    self.logger.error(f"Error processing character at frame {frame_idx}, obj_id {obj_id}: {e}")
                                    continue
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        self.logger.info(f"Processed {processed_count} frames, found {len(character_crops)} characters")
                    
                    # Clear GPU memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                frame_idx += 1
                
        cap.release()
        self.logger.info(f"Completed character extraction. Found {len(character_crops)} characters in {processed_count} frames")
        
        return character_crops
        
    def extract_character_from_image(self, image_path, output_dir=None, character_prompt=None):
        """
        Extract character from a single image using SAM
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save character crops (if None, don't save)
            character_prompt: Specific character to look for (e.g., "Little Prince", "woman in red dress")
            
        Returns:
            list: List of character crops (PIL Images)
        """
        try:
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Use character-specific prompt if provided
            prompt = character_prompt if character_prompt else "person"    
            self.logger.info(f"Extracting {prompt} from reference image: {image_path}")
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not read image: {image_path}")
                return []
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Initialize the segmentation tracker for character detection
            segtracker = SegTracker(self.segtracker_args, self.sam_args, self.aot_args)
            segtracker.restart_tracker()
            
            # Create directory for reference characters if needed
            if output_dir:
                ref_name = os.path.splitext(os.path.basename(image_path))[0]
                character_save_dir = os.path.join(output_dir, f"{ref_name}_character")
                os.makedirs(character_save_dir, exist_ok=True)
                self.logger.info(f"Saving reference character to: {character_save_dir}")
            else:
                character_save_dir = None
                
            # Detect character using SAM with specific character prompt
            with torch.cuda.amp.autocast():
                pred_mask, boxes = segtracker.detect_and_seg(
                    img_rgb,
                    prompt,
                    self.box_threshold,
                    self.text_threshold,
                    self.box_size_threshold,
                    self.reset_image
                )
            
            character_crops = []
            
            # Process detected characters
            if np.max(pred_mask) > 0:
                obj_ids = np.unique(pred_mask)
                obj_ids = obj_ids[obj_ids != 0]
                
                for obj_id in obj_ids:
                    # Create binary mask for this object
                    obj_mask = (pred_mask == obj_id)
                    
                    # Find bounding box for the mask
                    y_indices, x_indices = np.where(obj_mask)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        
                        # Check if the bounding box has valid dimensions
                        if x_max <= x_min or y_max <= y_min:
                            continue
                            
                        # Add some padding to the character crop
                        h, w = img_rgb.shape[:2]
                        pad = int(min(x_max-x_min, y_max-y_min) * 0.1)
                        x_min = max(0, x_min - pad)
                        y_min = max(0, y_min - pad)
                        x_max = min(w, x_max + pad)
                        y_max = min(h, y_max + pad)
                        
                        # Double-check the dimensions
                        if x_max <= x_min or y_max <= y_min or x_max > w or y_max > h:
                            continue
                            
                        # Crop the character
                        try:
                            character_crop = img_rgb[y_min:y_max, x_min:x_max]
                            
                            # Ensure the crop has valid dimensions
                            if character_crop.size == 0 or character_crop.shape[0] == 0 or character_crop.shape[1] == 0:
                                continue
                                
                            # Convert to PIL Image for the transformation pipeline
                            character_pil = Image.fromarray(character_crop)
                            
                            # Save character crop
                            if character_save_dir:
                                character_filename = f"character_ref_obj{obj_id:02d}.jpg"
                                character_path = os.path.join(character_save_dir, character_filename)
                                character_pil.save(character_path)
                                
                            character_crops.append(character_pil)
                        except Exception as e:
                            self.logger.error(f"Error processing reference character, obj_id {obj_id}: {e}")
                            continue
            
            # If no characters were detected, use the original image as fallback
            if not character_crops:
                self.logger.warning(f"No characters detected in reference image, using original image as fallback")
                try:
                    original_pil = Image.open(image_path).convert('RGB')
                    character_crops.append(original_pil)
                    
                    # Save the original as fallback character
                    if character_save_dir:
                        fallback_path = os.path.join(character_save_dir, "character_fallback.jpg")
                        original_pil.save(fallback_path)
                except Exception as e:
                    self.logger.error(f"Error using original image as fallback: {e}")
                        
            self.logger.info(f"Found {len(character_crops)} characters in reference image")
            return character_crops
            
        except Exception as e:
            self.logger.error(f"Error in character extraction from reference image: {e}")
            return []

    def extract_features(self, images, style):
        """
        Extract character embedding features using InceptionNext model
        
        Args:
            images: List of PIL Images
            style: 'cartoon' or 'real'
            
        Returns:
            torch.Tensor: Character embeddings
        """
        if not images:
            return None
            
        # Load the appropriate model
        model = self._load_model(style)
        
        # Preprocess images
        processed_images = torch.stack([self.transform(img) for img in images])
        processed_images = processed_images.to(self.device)
        
        # Extract features
        with torch.no_grad():
            embeddings = model(processed_images).detach().cpu()
            
        return embeddings

    def _delete_models(self):
        """Delete models to free up memory after evaluation"""
        if hasattr(self, 'cartoon_model') and self.cartoon_model is not None:
            del self.cartoon_model
            self.cartoon_model = None
        if hasattr(self, 'real_model') and self.real_model is not None:
            del self.real_model
            self.real_model = None
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Models deleted and memory cleared")

    def calculate_distances(self, ref_embedding, video_embeddings):
        """
        Calculate distances between reference character and video characters
        
        Args:
            ref_embedding: Embedding of reference character
            video_embeddings: Embeddings of characters from video
            
        Returns:
            dict: Distance metrics
        """
        if video_embeddings is None or ref_embedding is None:
            return {"error": "No characters detected in video or reference image"}
            
        # Calculate L2 distances using norm directly
        dists = [(ref_embedding - e2).norm().item() for e2 in video_embeddings]
        dists = torch.tensor(dists)
        
        # Calculate basic statistics
        avg_dist = torch.mean(dists).item()
        min_dist = torch.min(dists).item()
        max_dist = torch.max(dists).item()
        
        results = {
            "average_distance": avg_dist,
            "min_distance": min_dist,
            "max_distance": max_dist,
            "all_distances": dists.numpy().tolist()
        }
        
        return results

    def detect_style(self, image_path=None, video_path=None):
        """
        Detect if the input is cartoon style or real person style using Gemini API
        
        Args:
            image_path: Path to image file (optional)
            video_path: Path to video file (optional)
            
        Returns:
            str: 'cartoon' or 'real' or 'unknown'
        """
        if not self.gemini_api:
            self.logger.warning("Gemini API not available, defaulting to cartoon style")
            return 'cartoon'
        
        try:
            prompt = """Analyze this image/video and determine if it shows:
            1. Cartoon/animated characters (animated, drawn, CGI, stylized art style)
            2. Real people (photographic, realistic human characters)

            Please respond with only one word: 'cartoon' or 'real'"""
            
            if image_path:
                result = self.gemini_api.generate_from_images([image_path], prompt)
                self.logger.info(f"Style detection for image {image_path}: {result}")
            elif video_path:
                result = self.gemini_api.generate_from_videos([video_path], prompt)
                self.logger.info(f"Style detection for video {video_path}: {result}")
            else:
                return 'unknown'
            
            # Parse the result
            result_lower = result.lower().strip()
            if 'cartoon' in result_lower or 'animated' in result_lower:
                return 'cartoon'
            elif 'real' in result_lower or 'photographic' in result_lower:
                return 'real'
            else:
                self.logger.warning(f"Unclear style detection result: {result}, defaulting to cartoon")
                return 'cartoon'
                
        except Exception as e:
            self.logger.error(f"Error in style detection: {e}, defaulting to cartoon")
            return 'cartoon'

    def evaluate_video_image_character_similarity(self, video_path, image_path, output_dir=os.path.join(CURRENT_DIR, "tools", "Inceptionnext", "res", "character-sim"), sample_rate=16, character_prompt=None):
        """
        Evaluate character similarity between a video and a reference image with style detection
        """
        prompt_info = f" for {character_prompt}" if character_prompt else ""
        self.logger.info(f"Evaluating character distance{prompt_info} between video {video_path} and image {image_path}")
        
        # Detect style first
        style = self.detect_style(image_path=image_path)
        self.logger.info(f"Detected style: {style}")
        
        # Create output directory with consistent naming
        if output_dir:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            ref_name = os.path.splitext(os.path.basename(image_path))[0]
            char_folder = "_" + character_prompt.replace(" ", "_") if character_prompt else ""
            style_folder = f"_{style}"
            output_dir = os.path.join(output_dir, f"{timestamp}_{video_name}_vs_{ref_name}{char_folder}{style_folder}")
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Results will be saved to: {output_dir}")
        
        # Extract characters from video with character-specific prompt
        video_characters = self.extract_characters_from_video(video_path, output_dir, sample_rate, character_prompt)
        
        if not video_characters:
            self.logger.warning(f"No {character_prompt if character_prompt else 'characters'} detected in the video")
            return {"error": f"No {character_prompt if character_prompt else 'characters'} detected in video"}
            
        # Extract characters from reference image with character-specific prompt
        ref_characters = self.extract_character_from_image(image_path, output_dir, character_prompt)
        
        if not ref_characters:
            self.logger.warning(f"No {character_prompt if character_prompt else 'characters'} detected in the reference image")
            return {"error": f"No {character_prompt if character_prompt else 'characters'} detected in reference image"}
            
        # Extract features using the appropriate model based on style
        video_embeddings = self.extract_features(video_characters, style)
        ref_embedding = self.extract_features(ref_characters, style)[0]  # Take the first character if multiple
        
        # Calculate distances
        distance_results = self.calculate_distances(ref_embedding, video_embeddings)
        
        self.logger.info(f"Character distance results: Average={distance_results.get('average_distance', 'N/A')}")
        
        # Save results if output directory provided
        if output_dir:
            import json
            distance_results["style"] = style
            distance_results["model_used"] = f"{style}_character_model"
            with open(os.path.join(output_dir, "character_distance_results.json"), "w") as f:
                json.dump(distance_results, f, indent=4)
        
        return distance_results

    def evaluate_video_video_character_similarity(self, video_path1, video_path2, output_dir=os.path.join(CURRENT_DIR, "tools", "Inceptionnext", "res", "character-sim"), sample_rate=16, character_prompt=None):
        """
        Evaluate character similarity between two videos with style detection
        """
        prompt_info = f" for {character_prompt}" if character_prompt else ""
        self.logger.info(f"Evaluating character distance{prompt_info} between video {video_path1} and video {video_path2}")
        
        # Detect style from first video
        style = self.detect_style(video_path=video_path1)
        self.logger.info(f"Detected style: {style}")
        
        # Create output directory with consistent naming
        if output_dir:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            video1_name = os.path.splitext(os.path.basename(video_path1))[0]
            video2_name = os.path.splitext(os.path.basename(video_path2))[0]
            char_folder = "_" + character_prompt.replace(" ", "_") if character_prompt else ""
            style_folder = f"_{style}"
            output_dir = os.path.join(output_dir, f"{timestamp}_{video1_name}_vs_{video2_name}{char_folder}{style_folder}")
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Results will be saved to: {output_dir}")
        
        # Extract characters from both videos with character-specific prompt
        video1_characters = self.extract_characters_from_video(video_path1, 
                                                   output_dir + "/video1" if output_dir else None, 
                                                   sample_rate,
                                                   character_prompt)
        
        video2_characters = self.extract_characters_from_video(video_path2, 
                                                   output_dir + "/video2" if output_dir else None, 
                                                   sample_rate,
                                                   character_prompt)
        
        if not video1_characters:
            self.logger.warning(f"No {character_prompt if character_prompt else 'characters'} detected in the first video")
            return {"error": f"No {character_prompt if character_prompt else 'characters'} detected in first video"}
            
        if not video2_characters:
            self.logger.warning(f"No {character_prompt if character_prompt else 'characters'} detected in the second video")
            return {"error": f"No {character_prompt if character_prompt else 'characters'} detected in second video"}
            
        # Extract features using the appropriate model based on style
        video1_embeddings = self.extract_features(video1_characters, style)
        video2_embeddings = self.extract_features(video2_characters, style)
        
        # Calculate distances - comparing each character in video1 with each character in video2
        all_distances = []
        closest_distances = []
        
        for v1_emb in video1_embeddings:
            # Calculate distances from this character to all characters in video2
            character_distances = [(v1_emb - v2_emb).norm().item() for v2_emb in video2_embeddings]
            all_distances.extend(character_distances)
            # Find the closest match for each character in video1
            closest_distances.append(min(character_distances))
        
        # Calculate various statistics
        all_distances = torch.tensor(all_distances)
        closest_distances = torch.tensor(closest_distances)
        
        results = {
            "all_pairs": {
                "average_distance": torch.mean(all_distances).item(),
                "min_distance": torch.min(all_distances).item(),
                "max_distance": torch.max(all_distances).item(),
            },
            "closest_match": {
                "average_distance": torch.mean(closest_distances).item(),
                "min_distance": torch.min(closest_distances).item(),
                "max_distance": torch.max(closest_distances).item(),
            },
            "character_counts": {
                "video1_characters": len(video1_characters),
                "video2_characters": len(video2_characters),
            },
            "style": style,
            "model_used": f"{style}_character_model"
        }
        
        self.logger.info(f"Video-to-video character distance results:")
        self.logger.info(f"  All pairs - Avg: {results['all_pairs']['average_distance']:.4f}")
        self.logger.info(f"  Closest matches - Avg: {results['closest_match']['average_distance']:.4f}")
        
        # Save results if output directory provided
        if output_dir:
            import json
            with open(os.path.join(output_dir, "video_character_distance_results.json"), "w") as f:
                json.dump(results, f, indent=4)
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize with Gemini API keys
    CharacterEval = CharacterConsistencyBenchmark(
        gemini_api_keys=["YOUR_GEMINI_API_KEY", "YOUR_GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"],
        gemini_proxy="YOUR_PROXY_URL"
    )
    
    # Example 1: Cartoon character image-to-video comparison with automatic style detection
    image_video_results = CharacterEval.evaluate_video_image_character_similarity(
        video_path="your_video_path_here",
        image_path="your_reference_image_path_here",
        character_prompt="your_character_prompt_here"
    )
    print("Cartoon Character Image-to-Video Results:", image_video_results)
    
    # Example 2: Cartoon character video-to-video comparison
    video_video_results = CharacterEval.evaluate_video_video_character_similarity(
        video_path1="your_first_video_path_here",
        video_path2="your_second_video_path_here",
        character_prompt="your_character_prompt_here"
    )
    print("Cartoon Character Video-to-Video Results:", video_video_results)

    # Example 3: Real character image-to-video comparison with automatic style detection
    image_video_results = CharacterEval.evaluate_video_image_character_similarity(
        video_path="your_video_path_here",
        image_path="your_reference_image_path_here",
        character_prompt="your_character_prompt_here"
    )
    print("Real Character Image-to-Video Results:", image_video_results)
    
    # Example 4: Real character video-to-video comparison
    video_video_results = CharacterEval.evaluate_video_video_character_similarity(
        video_path1="your_first_video_path_here",
        video_path2="your_second_video_path_here",
        character_prompt="your_character_prompt_here"
    )
    print("Real Character Video-to-Video Results:", video_video_results)
