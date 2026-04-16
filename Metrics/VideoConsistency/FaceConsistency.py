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
from deepface import DeepFace

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

class IncepBenchmark:
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
        self.box_size_threshold = 0.35
        self.reset_image = True
        
        # Face similarity model setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize the InceptionNext model for face similarity (for cartoon style)
        self.face_model = create_model('inception_next_tiny', pretrained=False)
        checkpoint_path = os.path.join(CURRENT_DIR, "tools", "Inceptionnext", "output", "cartoonface.pth")
        load_checkpoint(self.face_model, checkpoint_path)
        self.face_model = self.face_model.to(self.device)
        self.face_model.eval()
        
        # Image transforms for face model
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.logger.info("IncepBenchmark initialized")

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
        
    def extract_faces_from_video(self, video_path, output_dir=None, sample_rate=16, character_prompt=None):
        """
        Extract faces from video frames using SAM and save them
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save face crops (if None, don't save)
            sample_rate: Process every Nth frame
            character_prompt: Specific character to look for (e.g., "face of Little Prince")
            
        Returns:
            list: List of face crops (PIL Images)
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Use character-specific prompt if provided
        prompt = character_prompt if character_prompt else "face"
        self.logger.info(f"Extracting {prompt} from video: {video_path}")
        face_crops = []
        
        # Initialize the segmentation tracker
        segtracker = SegTracker(self.segtracker_args, self.sam_args, self.aot_args)
        segtracker.restart_tracker()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(f"Video has {frame_count} frames at {fps} FPS")
        
        # Create a dedicated directory to save face crops
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if output_dir:
            face_save_dir = os.path.join(output_dir, f"{video_name}_faces")
            os.makedirs(face_save_dir, exist_ok=True)
            self.logger.info(f"Saving faces to: {face_save_dir}")
        else:
            face_save_dir = None
            
        frame_idx = 0
        processed_count = 0
        
        with torch.cuda.amp.autocast():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect face using SAM with specific character prompt if provided
                    pred_mask, boxes = segtracker.detect_and_seg(
                        frame_rgb, 
                        prompt, 
                        self.box_threshold, 
                        self.text_threshold,
                        self.box_size_threshold, 
                        self.reset_image
                    )
                    
                    if np.max(pred_mask) > 0:  # If a face is detected
                        # Process each detected object (potential face)
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
                                    
                                # Add some padding to the face crop
                                h, w = frame_rgb.shape[:2]
                                pad = int(min(x_max-x_min, y_max-y_min) * 0.1)
                                x_min = max(0, x_min - pad)
                                y_min = max(0, y_min - pad)
                                x_max = min(w, x_max + pad)
                                y_max = min(h, y_max + pad)
                                
                                # Double-check the dimensions
                                if x_max <= x_min or y_max <= y_min or x_max > w or y_max > h:
                                    continue
                                    
                                # Crop the face
                                try:
                                    face_crop = frame_rgb[y_min:y_max, x_min:x_max]
                                    
                                    # Ensure the crop has valid dimensions
                                    if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                                        continue
                                        
                                    # Convert to PIL Image for the transformation pipeline
                                    face_pil = Image.fromarray(face_crop)
                                    
                                    # Save face crop with unique filename
                                    if face_save_dir:
                                        face_filename = f"face_{frame_idx:05d}_obj{obj_id:02d}.jpg"
                                        face_path = os.path.join(face_save_dir, face_filename)
                                        face_pil.save(face_path)
                                        
                                    face_crops.append(face_pil)
                                except Exception as e:
                                    self.logger.error(f"Error processing face at frame {frame_idx}, obj_id {obj_id}: {e}")
                                    continue
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        self.logger.info(f"Processed {processed_count} frames, found {len(face_crops)} faces")
                    
                    # Clear GPU memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                frame_idx += 1
                
        cap.release()
        self.logger.info(f"Completed face extraction. Found {len(face_crops)} faces in {processed_count} frames")
        
        return face_crops
        
    def extract_face_from_image(self, image_path, output_dir=None, character_prompt=None):
        """
        Extract face from a single image using SAM
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save face crops (if None, don't save)
            character_prompt: Specific character to look for (e.g., "face of Little Prince")
            
        Returns:
            list: List of face crops (PIL Images)
        """
        try:
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Use character-specific prompt if provided
            prompt = character_prompt if character_prompt else "face"    
            self.logger.info(f"Extracting {prompt} from reference image: {image_path}")
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not read image: {image_path}")
                return []
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Initialize the segmentation tracker for face detection
            segtracker = SegTracker(self.segtracker_args, self.sam_args, self.aot_args)
            segtracker.restart_tracker()
            
            # Create directory for reference faces if needed
            if output_dir:
                ref_name = os.path.splitext(os.path.basename(image_path))[0]
                face_save_dir = os.path.join(output_dir, f"{ref_name}_face")
                os.makedirs(face_save_dir, exist_ok=True)
                self.logger.info(f"Saving reference face to: {face_save_dir}")
            else:
                face_save_dir = None
                
            # Detect face using SAM with specific character prompt
            with torch.cuda.amp.autocast():
                pred_mask, boxes = segtracker.detect_and_seg(
                    img_rgb,
                    prompt,
                    self.box_threshold,
                    self.text_threshold,
                    self.box_size_threshold,
                    self.reset_image
                )
            
            face_crops = []
            
            # Process detected faces
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
                            
                        # Add some padding to the face crop
                        h, w = img_rgb.shape[:2]
                        pad = int(min(x_max-x_min, y_max-y_min) * 0.1)
                        x_min = max(0, x_min - pad)
                        y_min = max(0, y_min - pad)
                        x_max = min(w, x_max + pad)
                        y_max = min(h, y_max + pad)
                        
                        # Double-check the dimensions
                        if x_max <= x_min or y_max <= y_min or x_max > w or y_max > h:
                            continue
                            
                        # Crop the face
                        try:
                            face_crop = img_rgb[y_min:y_max, x_min:x_max]
                            
                            # Ensure the crop has valid dimensions
                            if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                                continue
                                
                            # Convert to PIL Image for the transformation pipeline
                            face_pil = Image.fromarray(face_crop)
                            
                            # Save face crop
                            if face_save_dir:
                                face_filename = f"face_ref_obj{obj_id:02d}.jpg"
                                face_path = os.path.join(face_save_dir, face_filename)
                                face_pil.save(face_path)
                                
                            face_crops.append(face_pil)
                        except Exception as e:
                            self.logger.error(f"Error processing reference face, obj_id {obj_id}: {e}")
                            continue
            
            # If no faces were detected, use the original image as fallback
            if not face_crops:
                self.logger.warning(f"No faces detected in reference image, using original image as fallback")
                try:
                    original_pil = Image.open(image_path).convert('RGB')
                    face_crops.append(original_pil)
                    
                    # Save the original as fallback face
                    if face_save_dir:
                        fallback_path = os.path.join(face_save_dir, "face_fallback.jpg")
                        original_pil.save(fallback_path)
                except Exception as e:
                    self.logger.error(f"Error using original image as fallback: {e}")
                        
            self.logger.info(f"Found {len(face_crops)} faces in reference image")
            return face_crops
            
        except Exception as e:
            self.logger.error(f"Error in face extraction from reference image: {e}")
            return []

    def extract_features(self, images):
        """
        Extract face embedding features using InceptionNext model
        
        Args:
            images: List of PIL Images
            
        Returns:
            torch.Tensor: Face embeddings
        """
        if not images:
            return None
            
        # Preprocess images
        processed_images = torch.stack([self.transform(img) for img in images])
        processed_images = processed_images.to(self.device)
        
        # Extract features
        with torch.no_grad():
            embeddings = self.face_model(processed_images).detach().cpu()
            
        return embeddings

    def _delete_models(self):
        """Delete models to free up memory after evaluation"""
        if hasattr(self, 'face_model'):
            del self.face_model
            self.face_model = None
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Models deleted and memory cleared")

    def calculate_distances(self, ref_embedding, video_embeddings):
        """
        Calculate distances between reference face and video faces
        
        Args:
            ref_embedding: Embedding of reference face
            video_embeddings: Embeddings of faces from video
            
        Returns:
            dict: Distance metrics
        """
        if video_embeddings is None or ref_embedding is None:
            return {"error": "No faces detected in video or reference image"}
            
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
            2. Real people (photographic, realistic human faces)

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

    def evaluate_real_person_similarity(self, img1_path, img2_path):
        """
        Evaluate face similarity for real person style using DeepFace
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            dict: DeepFace verification results
        """
        try:
            self.logger.info(f"Using DeepFace for real person similarity: {img1_path} vs {img2_path}")
            result = DeepFace.verify(img1_path, img2_path)
            
            # Convert to our standard format
            distance_result = {
                "verified": result.get("verified", False),
                "distance": result.get("distance", 1.0),
                "threshold": result.get("threshold", 0.4),
                "model": result.get("model", "unknown"),
                "detector_backend": result.get("detector_backend", "unknown"),
                "similarity_score": 1.0 - result.get("distance", 1.0),  # Convert distance to similarity
            }
            
            self.logger.info(f"DeepFace result: verified={distance_result['verified']}, distance={distance_result['distance']:.4f}")
            return distance_result
            
        except Exception as e:
            self.logger.error(f"Error in DeepFace evaluation: {e}")
            return {"error": f"DeepFace evaluation failed: {str(e)}"}

    def evaluate_real_person_video_similarity(self, video_path, image_path=None, video_path2=None, output_dir=None, sample_rate=5):
        """
        Evaluate face similarity for real person style videos using DeepFace
        
        Args:
            video_path: Path to first video
            image_path: Path to reference image (for video-image comparison)
            video_path2: Path to second video (for video-video comparison)
            output_dir: Directory to save extracted frames
            sample_rate: Sample every Nth frame
            
        Returns:
            dict: Aggregated DeepFace results
        """
        try:
            # Create temp directory for frames if output_dir not provided
            if not output_dir:
                temp_frames_dir = os.path.join(BASE_DIR, "Evaluation", "results", "tmp", "video_consistency", "face_frames")
                os.makedirs(temp_frames_dir, exist_ok=True)
            else:
                temp_frames_dir = output_dir
                
            # Extract frames from first video
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create dedicated directory for video frames (similar to cartoon pipeline)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            if output_dir:
                video_frames_dir = os.path.join(output_dir, f"{video_name}_frames")
                os.makedirs(video_frames_dir, exist_ok=True)
                self.logger.info(f"Saving video frames to: {video_frames_dir}")
            else:
                video_frames_dir = temp_frames_dir
            
            frame_paths = []
            frame_idx = 0
            saved_frames = 0
            
            while cap.isOpened() and saved_frames < 10:  # Limit to 10 frames for efficiency
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % sample_rate == 0:
                    # Use similar naming convention as cartoon pipeline
                    frame_filename = f"frame_{frame_idx:05d}.jpg"
                    frame_path = os.path.join(video_frames_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    saved_frames += 1
                    self.logger.info(f"Saved frame {saved_frames}: {frame_filename}")
                    
                frame_idx += 1
            
            cap.release()
            
            if not frame_paths:
                return {"error": "No frames extracted from video"}
            
            # Compare with reference
            results = []
            if image_path:
                # Video-to-image comparison
                # Create directory for reference image if needed
                if output_dir:
                    ref_name = os.path.splitext(os.path.basename(image_path))[0]
                    ref_save_dir = os.path.join(output_dir, f"{ref_name}_reference")
                    os.makedirs(ref_save_dir, exist_ok=True)
                    # Copy reference image to output directory for consistency
                    import shutil
                    ref_copy_path = os.path.join(ref_save_dir, f"{ref_name}_ref.jpg")
                    shutil.copy2(image_path, ref_copy_path)
                    self.logger.info(f"Reference image copied to: {ref_copy_path}")
                
                for frame_path in frame_paths:
                    try:
                        result = self.evaluate_real_person_similarity(frame_path, image_path)
                        if "error" not in result:
                            results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Failed to compare frame {frame_path}: {e}")
                        continue
                        
            elif video_path2:
                # Video-to-video comparison - extract frames from second video too
                cap2 = cv2.VideoCapture(video_path2)
                
                # Create dedicated directory for second video frames
                video2_name = os.path.splitext(os.path.basename(video_path2))[0]
                if output_dir:
                    video2_frames_dir = os.path.join(output_dir, f"{video2_name}_frames")
                    os.makedirs(video2_frames_dir, exist_ok=True)
                    self.logger.info(f"Saving video2 frames to: {video2_frames_dir}")
                else:
                    video2_frames_dir = temp_frames_dir
                
                frame_paths2 = []
                frame_idx2 = 0
                saved_frames2 = 0
                
                while cap2.isOpened() and saved_frames2 < 10:
                    ret, frame = cap2.read()
                    if not ret:
                        break
                        
                    if frame_idx2 % sample_rate == 0:
                        # Use similar naming convention as cartoon pipeline
                        frame_filename = f"frame_{frame_idx2:05d}.jpg"
                        frame_path = os.path.join(video2_frames_dir, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        frame_paths2.append(frame_path)
                        saved_frames2 += 1
                        self.logger.info(f"Saved video2 frame {saved_frames2}: {frame_filename}")
                        
                    frame_idx2 += 1
                
                cap2.release()
                
                # Compare all frame pairs
                for frame_path1 in frame_paths:
                    for frame_path2 in frame_paths2:
                        try:
                            result = self.evaluate_real_person_similarity(frame_path1, frame_path2)
                            if "error" not in result:
                                results.append(result)
                        except Exception as e:
                            continue
            
            # Clean up temporary frames only if we created a temp directory
            if not output_dir:
                import shutil
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
            
            if not results:
                return {"error": "No successful comparisons"}
            
            # Aggregate results
            distances = [r["distance"] for r in results]
            verified_count = sum(1 for r in results if r["verified"])
            
            aggregated = {
                "average_distance": np.mean(distances),
                "min_distance": np.min(distances),
                "max_distance": np.max(distances),
                "verification_rate": verified_count / len(results),
                "total_comparisons": len(results),
                "verified_matches": verified_count,
                "frame_counts": {
                    "video1_frames": len(frame_paths),
                    "video2_frames": len(frame_paths2) if video_path2 else 0,
                },
                "style": "real"
            }
            
            self.logger.info(f"DeepFace video results: avg_distance={aggregated['average_distance']:.4f}, verification_rate={aggregated['verification_rate']:.2f}")
            
            # Save results if output directory provided (similar to cartoon pipeline)
            if output_dir:
                import json
                results_filename = "deepface_results.json"
                results_path = os.path.join(output_dir, results_filename)
                with open(results_path, "w") as f:
                    json.dump(aggregated, f, indent=4)
                self.logger.info(f"Results saved to: {results_path}")
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error in real person video evaluation: {e}")
            return {"error": f"Real person video evaluation failed: {str(e)}"}

    def evaluate_video_image_face_similarity(self, video_path, image_path, output_dir=os.path.join(CURRENT_DIR, "tools", "Inceptionnext", "res", "face-sim"), sample_rate=16, character_prompt=None):
        """
        Evaluate face similarity between a video and a reference image with style detection
        """
        prompt_info = f" for {character_prompt}" if character_prompt else ""
        self.logger.info(f"Evaluating face distance{prompt_info} between video {video_path} and image {image_path}")
        
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
        
        if style == 'real':
            # Use DeepFace for real person style
            return self.evaluate_real_person_video_similarity(video_path, image_path=image_path, output_dir=output_dir, sample_rate=sample_rate)
        else:
            # Use existing cartoon pipeline
            # Extract faces from video with character-specific prompt
            video_faces = self.extract_faces_from_video(video_path, output_dir, sample_rate, character_prompt)
            
            if not video_faces:
                self.logger.warning(f"No {character_prompt if character_prompt else 'faces'} detected in the video")
                return {"error": f"No {character_prompt if character_prompt else 'faces'} detected in video"}
                
            # Extract faces from reference image with character-specific prompt
            ref_faces = self.extract_face_from_image(image_path, output_dir, character_prompt)
            
            if not ref_faces:
                self.logger.warning(f"No {character_prompt if character_prompt else 'faces'} detected in the reference image")
                return {"error": f"No {character_prompt if character_prompt else 'faces'} detected in reference image"}
                
            # Extract features
            video_embeddings = self.extract_features(video_faces)
            ref_embedding = self.extract_features(ref_faces)[0]  # Take the first face if multiple
            
            # Calculate distances
            distance_results = self.calculate_distances(ref_embedding, video_embeddings)
            
            self.logger.info(f"Face distance results: Average={distance_results.get('average_distance', 'N/A')}")
            
            # Save results if output directory provided
            if output_dir:
                import json
                distance_results["style"] = style
                with open(os.path.join(output_dir, "face_distance_results.json"), "w") as f:
                    json.dump(distance_results, f, indent=4)
            
            return distance_results

    def evaluate_video_video_face_similarity(self, video_path1, video_path2, output_dir=os.path.join(CURRENT_DIR, "tools", "Inceptionnext", "res", "face-sim"), sample_rate=16, character_prompt=None):
        """
        Evaluate face similarity between two videos with style detection
        """
        prompt_info = f" for {character_prompt}" if character_prompt else ""
        self.logger.info(f"Evaluating face distance{prompt_info} between video {video_path1} and video {video_path2}")
        
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
        
        if style == 'real':
            # Use DeepFace for real person style
            return self.evaluate_real_person_video_similarity(video_path1, video_path2=video_path2, output_dir=output_dir, sample_rate=sample_rate)
        else:
            # Use existing cartoon pipeline
            # Extract faces from both videos with character-specific prompt
            video1_faces = self.extract_faces_from_video(video_path1, 
                                                       output_dir + "/video1" if output_dir else None, 
                                                       sample_rate,
                                                       character_prompt)
            
            video2_faces = self.extract_faces_from_video(video_path2, 
                                                       output_dir + "/video2" if output_dir else None, 
                                                       sample_rate,
                                                       character_prompt)
            
            if not video1_faces:
                self.logger.warning(f"No {character_prompt if character_prompt else 'faces'} detected in the first video")
                return {"error": f"No {character_prompt if character_prompt else 'faces'} detected in first video"}
                
            if not video2_faces:
                self.logger.warning(f"No {character_prompt if character_prompt else 'faces'} detected in the second video")
                return {"error": f"No {character_prompt if character_prompt else 'faces'} detected in second video"}
                
            # Extract features
            video1_embeddings = self.extract_features(video1_faces)
            video2_embeddings = self.extract_features(video2_faces)
            
            # Calculate distances - comparing each face in video1 with each face in video2
            all_distances = []
            closest_distances = []
            
            for v1_emb in video1_embeddings:
                # Calculate distances from this face to all faces in video2
                face_distances = [(v1_emb - v2_emb).norm().item() for v2_emb in video2_embeddings]
                all_distances.extend(face_distances)
                # Find the closest match for each face in video1
                closest_distances.append(min(face_distances))
            
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
                "face_counts": {
                    "video1_faces": len(video1_faces),
                    "video2_faces": len(video2_faces),
                },
                "style": style
            }
            
            self.logger.info(f"Video-to-video face distance results:")
            self.logger.info(f"  All pairs - Avg: {results['all_pairs']['average_distance']:.4f}")
            self.logger.info(f"  Closest matches - Avg: {results['closest_match']['average_distance']:.4f}")
            
            # Save results if output directory provided
            if output_dir:
                import json
                with open(os.path.join(output_dir, "video_face_distance_results.json"), "w") as f:
                    json.dump(results, f, indent=4)
            
            return results

# Example usage
if __name__ == "__main__":
    # Initialize with Gemini API keys
    IncepEval = IncepBenchmark(
        gemini_api_keys=["YOUR_GEMINI_API_KEY", "YOUR_GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"],
        gemini_proxy="YOUR_PROXY_URL"
    )
    
    # Example 1: Cartoon Image-to-video comparison with automatic style detection
    image_video_results = IncepEval.evaluate_video_image_face_similarity(
        video_path="your_video_path_here",
        image_path="your_reference_image_path_here",
        character_prompt="your_character_prompt_here"
    )
    print("Cartoon Image-to-Video Results:", image_video_results)
    
    # Example 2: Cartoon Video-to-video comparison with character-specific prompt
    video_video_results = IncepEval.evaluate_video_video_face_similarity(
        video_path1="your_first_video_path_here",
        video_path2="your_second_video_path_here",
        character_prompt="your_character_prompt_here"
    )
    print("Cartoon Video-to-Video Results:", video_video_results)

    # Example 3: Cartoon Image-to-video comparison with automatic style detection
    image_video_results = IncepEval.evaluate_video_image_face_similarity(
        video_path="your_video_path_here",
        image_path="your_reference_image_path_here",
        character_prompt="your_character_prompt_here"
    )
    print("Real Image-to-Video Results:", image_video_results)
    
    # Example 4: Cartoon Video-to-video comparison with character-specific prompt
    video_video_results = IncepEval.evaluate_video_video_face_similarity(
        video_path1="your_first_video_path_here",
        video_path2="your_second_video_path_here",
        character_prompt="your_character_prompt_here"
    )
    print("Real Video-to-Video Results:", video_video_results)

