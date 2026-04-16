import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import importlib

# Lazy-load dependencies to avoid heavy top-level imports

# Prepare Gemini import path without importing the module immediately
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(BASE_DIR, "Tools"))


class VideoConsistencyEvaluator:
    def __init__(self, gemini_api_keys=None):
        """Initialize in lazy-loading mode and load each sub-metric module on demand."""
        print("Initializing Video Consistency Evaluator (lazy-loading mode)...")

        # Keep Gemini credentials/proxy without creating a client immediately
        self.gemini_api_keys = gemini_api_keys
        self.gemini_proxy = "YOUR_PROXY_URL"
        self.gemini_api = None

        # Placeholder evaluators, initialized only when needed
        self.character_evaluator = None
        self.face_evaluator = None
        self.background_evaluator = None
        self.clothes_evaluator = None
        self.size_evaluator = None

        print("Evaluators will be loaded only when their metrics run.\n")

    # ---------------------------
    # Lazy ensure helpers
    # ---------------------------
    def _ensure_character_evaluator(self):
        if self.character_evaluator is None:
            try:
                module = importlib.import_module('CharacterConsistency')
                cls = getattr(module, 'CharacterConsistencyBenchmark')
                self.character_evaluator = cls(
                    gemini_api_keys=self.gemini_api_keys,
                    gemini_proxy=self.gemini_proxy
                )
                print("✓ Character consistency evaluator loaded (lazy)")
            except Exception as e:
                raise RuntimeError(f"Failed to load CharacterConsistencyBenchmark: {e}")

    def _ensure_face_evaluator(self):
        if self.face_evaluator is None:
            try:
                module = importlib.import_module('FaceConsistency')
                cls = getattr(module, 'IncepBenchmark')
                self.face_evaluator = cls(
                    gemini_api_keys=self.gemini_api_keys,
                    gemini_proxy=self.gemini_proxy
                )
                print("✓ Face consistency evaluator loaded (lazy)")
            except Exception as e:
                raise RuntimeError(f"Failed to load IncepBenchmark: {e}")

    def _ensure_background_evaluator(self):
        if self.background_evaluator is None:
            try:
                module = importlib.import_module('BackgroundConsistency')
                cls = getattr(module, 'BackgroundConsistencyProcessor')
                self.background_evaluator = cls()
                print("✓ Background consistency evaluator loaded (lazy)")
            except Exception as e:
                raise RuntimeError(f"Failed to load BackgroundConsistencyProcessor: {e}")

    def _ensure_clothes_evaluator(self):
        if self.clothes_evaluator is None:
            try:
                module = importlib.import_module('ClothesColorConsistency')
                cls = getattr(module, 'ClothesColorConsistency')
                self.clothes_evaluator = cls(api_keys=self.gemini_api_keys, proxy=self.gemini_proxy)
                print("✓ Clothes color consistency evaluator loaded (lazy)")
            except Exception as e:
                raise RuntimeError(f"Failed to load ClothesColorConsistency: {e}")

    def _ensure_size_evaluator(self):
        if self.size_evaluator is None:
            try:
                # Ensure Gemini client first
                if self.gemini_api is None:
                    gemini_module = importlib.import_module('gemini_api')
                    GeminiAPI = getattr(gemini_module, 'GeminiAPI')
                    self.gemini_api = GeminiAPI(api_keys=self.gemini_api_keys, proxy=self.gemini_proxy)
                    print("✓ Gemini API client loaded (lazy)")

                module = importlib.import_module('RelativeSizeConsistency')
                cls = getattr(module, 'RelativeSizeConsistency')
                self.size_evaluator = cls(self.gemini_api)
                print("✓ Relative size consistency evaluator loaded (lazy)")
            except Exception as e:
                raise RuntimeError(f"Failed to load RelativeSizeConsistency: {e}")

    def load_existing_results(self, output_file: str) -> Dict[str, Any]:
        """Load existing results from JSON file if it exists"""
        if not output_file or not os.path.exists(output_file):
            return {}
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"✓ Loaded existing results from: {output_file}")
            return results
        except Exception as e:
            print(f"⚠️  Error loading existing results: {e}")
            return {}

    def is_evaluation_complete(self, existing_results: Dict[str, Any], evaluation_type: str, 
                              video_files: List[str] = None, video_characters: Dict[str, List[Dict[str, str]]] = None) -> bool:
        """Check if a specific evaluation is already complete"""
        if not existing_results or evaluation_type not in existing_results:
            return False
        
        eval_results = existing_results[evaluation_type]
        if self._contains_resource_exhausted(eval_results):
            print(f"Detected 429 RESOURCE_EXHAUSTED in historical {evaluation_type} results, re-running evaluation")
            return False
        
        # Check if summary exists (indicates completion)
        if "summary" not in eval_results:
            return False
        
        # For evaluations that compare video pairs, check if all pairs are present
        if evaluation_type in ["character_consistency", "face_consistency", "background_consistency", 
                              "clothes_color_consistency", "relative_size_consistency"]:
            
            if evaluation_type in ["background_consistency", "clothes_color_consistency", "relative_size_consistency"]:
                # These only do video-to-video comparisons
                if "video_pairs" not in eval_results:
                    return False
                return True
            
            elif evaluation_type in ["character_consistency", "face_consistency"]:
                # These do both video-to-video and video-to-reference
                if "video_to_video" not in eval_results or "video_to_reference" not in eval_results:
                    return False
                return True
        
        return True

    def check_video_files_match(self, existing_results: Dict[str, Any], video_files: List[str]) -> bool:
        """Check if the video files in existing results match current video files"""
        if not existing_results or "video_files" not in existing_results:
            return False
        
        existing_video_files = existing_results["video_files"]
        current_video_files = [os.path.basename(f) for f in video_files]
        
        return set(existing_video_files) == set(current_video_files)

    def merge_results(self, existing_results: Dict[str, Any], new_results: Dict[str, Any], 
                     completed_evaluations: List[str]) -> Dict[str, Any]:
        """Merge existing results with new results, keeping completed evaluations from existing"""
        merged_results = new_results.copy()
        
        for eval_type in completed_evaluations:
            if eval_type in existing_results:
                merged_results[eval_type] = existing_results[eval_type]
                print(f"✓ Reused existing {eval_type} results")
        
        # Update summary with reused results
        if "summary" in existing_results:
            if "summary" not in merged_results:
                merged_results["summary"] = {}
            
            for eval_type in completed_evaluations:
                if eval_type == "character_consistency":
                    if "character_v2v_distance" in existing_results["summary"]:
                        merged_results["summary"]["character_v2v_distance"] = existing_results["summary"]["character_v2v_distance"]
                    if "character_v2r_distance" in existing_results["summary"]:
                        merged_results["summary"]["character_v2r_distance"] = existing_results["summary"]["character_v2r_distance"]
                elif eval_type == "face_consistency":
                    if "face_v2v_distance" in existing_results["summary"]:
                        merged_results["summary"]["face_v2v_distance"] = existing_results["summary"]["face_v2v_distance"]
                    if "face_v2r_distance" in existing_results["summary"]:
                        merged_results["summary"]["face_v2r_distance"] = existing_results["summary"]["face_v2r_distance"]
                elif eval_type == "background_consistency":
                    if "background_avg_distance" in existing_results["summary"]:
                        merged_results["summary"]["background_avg_distance"] = existing_results["summary"]["background_avg_distance"]
                elif eval_type == "clothes_color_consistency":
                    if "clothes_avg_score" in existing_results["summary"]:
                        merged_results["summary"]["clothes_avg_score"] = existing_results["summary"]["clothes_avg_score"]
                elif eval_type == "relative_size_consistency":
                    if "size_avg_consistency" in existing_results["summary"]:
                        merged_results["summary"]["size_avg_consistency"] = existing_results["summary"]["size_avg_consistency"]
        
        return merged_results

    def get_video_files(self, video_dir: str) -> List[str]:
        """Get all video files from directory in alphabetical order"""
        if not os.path.exists(video_dir):
            raise ValueError(f"Video directory does not exist: {video_dir}")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
        video_files = []
        
        for file in os.listdir(video_dir):
            if Path(file).suffix.lower() in video_extensions:
                if 'LongLive' in video_dir or 'complete' not in file.lower():
                    video_files.append(os.path.join(video_dir, file))
        
        if not video_files:
            raise ValueError(f"No video files found in directory: {video_dir}")
        
        # Sort alphabetically
        video_files.sort()
        print(f"Found {len(video_files)} video files:")
        for i, video in enumerate(video_files):
            print(f"  {i+1}. {os.path.basename(video)}")
        print()
        
        return video_files

    def load_script(self, script_path: str) -> Dict[str, Any]:
        """Load and parse the script.json file"""
        if not os.path.exists(script_path):
            raise ValueError(f"Script file does not exist: {script_path}")
        
        with open(script_path, 'r', encoding='utf-8') as f:
            script_data = json.load(f)
        
        print(f"Loaded script with {len(script_data.get('characters', []))} characters")
        print(f"Script contains {len(script_data.get('script', []))} scenes")
        
        return script_data

    def get_character_ref_images(self, characters_dir: str, script_data: Dict[str, Any]) -> Dict[str, str]:
        """Get reference images for all characters"""
        character_refs = {}
        characters = script_data.get('characters', [])
        
        for character in characters:
            char_name = character['name']
            # Replace spaces with underscores for directory name
            char_dir_name = char_name.replace(' ', '_')
            char_dir = os.path.join(characters_dir, char_dir_name)
            
            if os.path.exists(char_dir):
                # Look for img.jpg or other image files
                possible_files = ['img.jpg', 'img.png', 'image.jpg', 'image.png', 'ref.jpg', 'ref.png']
                
                for file_name in possible_files:
                    img_path = os.path.join(char_dir, file_name)
                    if os.path.exists(img_path):
                        character_refs[char_name] = img_path
                        print(f"  Found reference for {char_name}: {file_name}")
                        break
                else:
                    # If no standard file found, take the first image file
                    for file in os.listdir(char_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            character_refs[char_name] = os.path.join(char_dir, file)
                            print(f"  Found reference for {char_name}: {file}")
                            break
            
            if char_name not in character_refs:
                print(f"  ⚠️  No reference image found for {char_name}")
        
        return character_refs

    def analyze_video_characters(self, video_files: List[str], script_data: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """Analyze which characters should appear in each video based on script"""
        print("Analyzing video-character mappings...")
        
        # Initialize the result dictionary
        video_characters = {}
        
        # Create mapping from clip names to character lists
        clip_to_characters = {}
        scenes = script_data.get('script', [])
        
        clip_counter = 1  # Start from 1 for sequential numbering
        
        for scene in scenes:
            scene_id = scene.get('scene')
            clips = scene.get('clips', [])
            
            for clip in clips:
                clip_id = clip.get('id')
                character_ids = clip.get('characters', [])
                
                # Create clip name using sequential numbering (01, 02, 03...)
                clip_name = f"{clip_counter:02d}"
                
                # Get character info from IDs
                character_info = []
                for char_id in character_ids:
                    for character in script_data.get('characters', []):
                        if str(character['id']) == str(char_id):
                            character_info.append({
                                'name': character['name'],
                                'short': character.get('short', character['name'])  # Use short field if available, fallback to name
                            })
                
                clip_to_characters[clip_name] = character_info
                clip_counter += 1  # Increment for next clip
        
        # Match video files to clips
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Try to match video name to clip name
            matched_characters = []
            for clip_name, characters in clip_to_characters.items():
                if clip_name.lower() in video_name.lower() or video_name.lower() in clip_name.lower():
                    matched_characters = characters
                    break
            
            video_characters[video_name] = matched_characters
            if matched_characters:
                char_names = [char['name'] for char in matched_characters]
                print(f"  {video_name}: {char_names}")
            else:
                print(f"  {video_name}: No character mapping found")
        
        # Ensure we return the correct video_characters dictionary, not script_data
        print(f"Returning video_characters with {len(video_characters)} video mappings")
        return video_characters

    def evaluate_character_consistency(self, video_files: List[str], character_refs: Dict[str, str], 
                                     video_characters: Dict[str, List[Dict[str, str]]], output_dir: str = None) -> Dict[str, Any]:
        """Evaluate character consistency across videos and against references"""
        print("=== Evaluating Character Consistency ===")

        # Ensure evaluator is available
        self._ensure_character_evaluator()
        
        results = {
            "video_to_video": {},
            "video_to_reference": {},
            "summary": {}
        }
        
        if output_dir:
            char_output_dir = os.path.join(output_dir, "character_consistency")
            os.makedirs(char_output_dir, exist_ok=True)
        else:
            char_output_dir = None
        
        # 1. Video-to-video comparison (adjacent pairs)
        print("Evaluating video-to-video character consistency...")
        for i in range(len(video_files) - 1):
            video1_path = video_files[i]
            video2_path = video_files[i + 1]
            video1_name = os.path.splitext(os.path.basename(video1_path))[0]
            video2_name = os.path.splitext(os.path.basename(video2_path))[0]
            
            # Get common characters between the two videos
            chars1_dict = {char['name']: char for char in video_characters.get(video1_name, [])}
            chars2_dict = {char['name']: char for char in video_characters.get(video2_name, [])}
            common_char_names = set(chars1_dict.keys()).intersection(set(chars2_dict.keys()))
            
            pair_name = f"{video1_name} -> {video2_name}"
            print(f"Comparing {pair_name}")
            print(f"  Common characters: {list(common_char_names)}")
            
            pair_results = {}
            
            for char_name in common_char_names:
                # try:
                # Use the short description as prompt
                char_short = chars1_dict[char_name]['short']
                print(f"Comparing {char_name} ({char_short})")
                result = self.character_evaluator.evaluate_video_video_character_similarity(
                    video1_path, video2_path,
                    output_dir=char_output_dir,
                    character_prompt=char_short
                )
                pair_results[char_name] = result
                print(f"    {char_name} ({char_short}): avg_distance={result.get('closest_match', {}).get('average_distance', 'N/A')}")
                # except Exception as e:
                #     print(f"    {char_name}: Error - {e}")
                #     pair_results[char_name] = {"error": str(e)}
            
            results["video_to_video"][pair_name] = pair_results
        
        # 2. Video-to-reference comparison
        print("\nEvaluating video-to-reference character consistency...")
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_chars = video_characters.get(video_name, [])
            
            print(f"Comparing {video_name} to references")
            video_results = {}
            
            for char_info in video_chars:
                char_name = char_info['name']
                char_short = char_info['short']
                
                if char_name in character_refs:
                    try:
                        result = self.character_evaluator.evaluate_video_image_character_similarity(
                            video_path, character_refs[char_name],
                            output_dir=char_output_dir,
                            character_prompt=char_short
                        )
                        video_results[char_name] = result
                        print(f"    {char_name} ({char_short}): avg_distance={result.get('average_distance', 'N/A')}")
                    except Exception as e:
                        print(f"    {char_name}: Error - {e}")
                        video_results[char_name] = {"error": str(e)}
                else:
                    print(f"    {char_name}: No reference image available")
            
            results["video_to_reference"][video_name] = video_results
        
        # Calculate summary statistics
        all_v2v_distances = []
        all_v2r_distances = []
        
        for pair_results in results["video_to_video"].values():
            for char_result in pair_results.values():
                if "error" not in char_result and "closest_match" in char_result:
                    all_v2v_distances.append(char_result["closest_match"]["average_distance"])
        
        for video_results in results["video_to_reference"].values():
            for char_result in video_results.values():
                if "error" not in char_result and "average_distance" in char_result:
                    all_v2r_distances.append(char_result["average_distance"])
        
        results["summary"] = {
            "avg_video_to_video_distance": float(np.mean(all_v2v_distances)) if all_v2v_distances else 0.0,
            "avg_video_to_reference_distance": float(np.mean(all_v2r_distances)) if all_v2r_distances else 0.0,
            "total_v2v_comparisons": int(len(all_v2v_distances)),
            "total_v2r_comparisons": int(len(all_v2r_distances))
        }
        
        print(f"Character consistency summary:")
        print(f"  Avg video-to-video distance: {results['summary']['avg_video_to_video_distance']:.4f}")
        print(f"  Avg video-to-reference distance: {results['summary']['avg_video_to_reference_distance']:.4f}")
        
        return results

    def evaluate_face_consistency(self, video_files: List[str], character_refs: Dict[str, str], 
                                video_characters: Dict[str, List[Dict[str, str]]], output_dir: str = None) -> Dict[str, Any]:
        """Evaluate face consistency across videos and against references"""
        print("=== Evaluating Face Consistency ===")

        # Ensure evaluator is available
        self._ensure_face_evaluator()
        
        results = {
            "video_to_video": {},
            "video_to_reference": {},
            "summary": {}
        }
        
        if output_dir:
            face_output_dir = os.path.join(output_dir, "face_consistency")
            os.makedirs(face_output_dir, exist_ok=True)
        else:
            face_output_dir = None
        
        # 1. Video-to-video comparison (adjacent pairs)
        print("Evaluating video-to-video face consistency...")
        for i in range(len(video_files) - 1):
            video1_path = video_files[i]
            video2_path = video_files[i + 1]
            video1_name = os.path.splitext(os.path.basename(video1_path))[0]
            video2_name = os.path.splitext(os.path.basename(video2_path))[0]
            
            # Get common characters between the two videos
            chars1_dict = {char['name']: char for char in video_characters.get(video1_name, [])}
            chars2_dict = {char['name']: char for char in video_characters.get(video2_name, [])}
            common_char_names = set(chars1_dict.keys()).intersection(set(chars2_dict.keys()))
            
            pair_name = f"{video1_name} -> {video2_name}"
            print(f"Comparing {pair_name}")
            
            pair_results = {}
            
            for char_name in common_char_names:
                try:
                    # Use the short description for face prompt
                    char_short = chars1_dict[char_name]['short']
                    face_prompt = f"{char_short}"
                    
                    result = self.face_evaluator.evaluate_video_video_face_similarity(
                        video1_path, video2_path,
                        output_dir=face_output_dir,
                        character_prompt=face_prompt
                    )
                    pair_results[char_name] = result
                    
                    # Extract average distance based on result format
                    avg_dist = "N/A"
                    if "error" not in result:
                        if "closest_match" in result:
                            avg_dist = result["closest_match"]["average_distance"]
                        elif "average_distance" in result:
                            avg_dist = result["average_distance"]
                    
                    print(f"    {char_name} ({face_prompt}): avg_distance={avg_dist}")
                except Exception as e:
                    print(f"    {char_name}: Error - {e}")
                    pair_results[char_name] = {"error": str(e)}
            
            results["video_to_video"][pair_name] = pair_results
        
        # 2. Video-to-reference comparison
        print("\nEvaluating video-to-reference face consistency...")
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_chars = video_characters.get(video_name, [])
            
            print(f"Comparing {video_name} to references")
            video_results = {}
            
            for char_info in video_chars:
                char_name = char_info['name']
                char_short = char_info['short']
                face_prompt = f"{char_short}"
                
                if char_name in character_refs:
                    try:
                        result = self.face_evaluator.evaluate_video_image_face_similarity(
                            video_path, character_refs[char_name],
                            output_dir=face_output_dir,
                            character_prompt=face_prompt
                        )
                        video_results[char_name] = result
                        avg_dist = result.get('average_distance', 'N/A')
                        print(f"    {char_name} ({face_prompt}): avg_distance={avg_dist}")
                    except Exception as e:
                        print(f"    {char_name}: Error - {e}")
                        video_results[char_name] = {"error": str(e)}
                else:
                    print(f"    {char_name}: No reference image available")
            
            results["video_to_reference"][video_name] = video_results
        
        # Calculate summary statistics
        all_v2v_distances = []
        all_v2r_distances = []
        
        for pair_results in results["video_to_video"].values():
            for char_result in pair_results.values():
                if "error" not in char_result:
                    if "closest_match" in char_result:
                        all_v2v_distances.append(char_result["closest_match"]["average_distance"])
                    elif "average_distance" in char_result:
                        all_v2v_distances.append(char_result["average_distance"])
        
        for video_results in results["video_to_reference"].values():
            for char_result in video_results.values():
                if "error" not in char_result and "average_distance" in char_result:
                    all_v2r_distances.append(char_result["average_distance"])
        
        results["summary"] = {
            "avg_video_to_video_distance": float(np.mean(all_v2v_distances)) if all_v2v_distances else 0.0,
            "avg_video_to_reference_distance": float(np.mean(all_v2r_distances)) if all_v2r_distances else 0.0,
            "total_v2v_comparisons": int(len(all_v2v_distances)),
            "total_v2r_comparisons": int(len(all_v2r_distances))
        }
        
        print(f"Face consistency summary:")
        print(f"  Avg video-to-video distance: {results['summary']['avg_video_to_video_distance']:.4f}")
        print(f"  Avg video-to-reference distance: {results['summary']['avg_video_to_reference_distance']:.4f}")
        
        return results

    def evaluate_background_consistency(self, video_files: List[str], output_dir: str = None) -> Dict[str, Any]:
        """Evaluate background consistency between adjacent video pairs"""
        print("=== Evaluating Background Consistency ===")

        # Ensure evaluator is available
        self._ensure_background_evaluator()
        
        results = {
            "video_pairs": {},
            "summary": {}
        }
        
        if output_dir:
            bg_output_dir = os.path.join(output_dir, "background_consistency")
            os.makedirs(bg_output_dir, exist_ok=True)
        else:
            bg_output_dir = None
        
        distances = []
        
        # Compare adjacent video pairs
        for i in range(len(video_files) - 1):
            video1_path = video_files[i]
            video2_path = video_files[i + 1]
            video1_name = os.path.splitext(os.path.basename(video1_path))[0]
            video2_name = os.path.splitext(os.path.basename(video2_path))[0]
            
            pair_name = f"{video1_name} -> {video2_name}"
            print(f"Comparing background: {pair_name}")
            
            try:
                # Extract representative frames from videos for background comparison
                # Use middle frame from each video
                cap1 = cv2.VideoCapture(video1_path)
                cap2 = cv2.VideoCapture(video2_path)
                
                frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Get middle frames
                cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_count1 // 2)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_count2 // 2)
                
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                cap1.release()
                cap2.release()
                
                if ret1 and ret2:
                    # Save frames temporarily
                    temp_dir = os.path.join(BASE_DIR, "Evaluation", "results", "tmp", "background_frames") if not bg_output_dir else bg_output_dir
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    frame1_path = os.path.join(temp_dir, f"{video1_name}_frame.jpg")
                    frame2_path = os.path.join(temp_dir, f"{video2_name}_frame.jpg")
                    
                    cv2.imwrite(frame1_path, frame1)
                    cv2.imwrite(frame2_path, frame2)
                    
                    # Calculate background similarity
                    distance = self.background_evaluator.process_images_and_calculate_similarity(
                        frame1_path, frame2_path
                    )
                    
                    results["video_pairs"][pair_name] = {
                        "distance": distance,
                        "frame1": frame1_path,
                        "frame2": frame2_path
                    }
                    distances.append(distance)
                    print(f"  Background distance: {distance:.4f}")
                    
                else:
                    results["video_pairs"][pair_name] = {"error": "Could not extract frames"}
                    print(f"  Error: Could not extract frames")
                
            except Exception as e:
                results["video_pairs"][pair_name] = {"error": str(e)}
                print(f"  Error: {e}")
        
        # Calculate summary
        results["summary"] = {
            "average_distance": float(np.mean(distances)) if distances else 0.0,
            "min_distance": float(np.min(distances)) if distances else 0.0,
            "max_distance": float(np.max(distances)) if distances else 0.0,
            "total_comparisons": int(len(distances))
        }
        
        print(f"Background consistency summary:")
        print(f"  Avg distance: {results['summary']['average_distance']:.4f}")
        print(f"  Min distance: {results['summary']['min_distance']:.4f}")
        print(f"  Max distance: {results['summary']['max_distance']:.4f}")
        
        return results

    def evaluate_clothes_color_consistency(self, video_files: List[str], video_characters: Dict[str, List[Dict[str, str]]], output_dir: str = None) -> Dict[str, Any]:
        """Evaluate clothes color consistency - for each video, find the nearest subsequent video with the same character"""
        print("=== Evaluating Clothes Color Consistency ===")

        # Ensure evaluator is available
        self._ensure_clothes_evaluator()
        
        results = {
            "video_pairs": {},
            "summary": {}
        }
        
        if output_dir:
            clothes_output_dir = os.path.join(output_dir, "clothes_consistency")
            os.makedirs(clothes_output_dir, exist_ok=True)
        else:
            clothes_output_dir = os.path.join(BASE_DIR, "Evaluation", "results", "tmp", "clothes_frames")
        
        os.makedirs(clothes_output_dir, exist_ok=True)
        
        all_scores = []
        
        # For each video, search forward for the nearest video with common characters
        for i in range(len(video_files)):
            video1_path = video_files[i]
            video1_name = os.path.splitext(os.path.basename(video1_path))[0]
            chars1_dict = {char['name']: char for char in video_characters.get(video1_name, [])}
            
            # Search forward for the nearest video with common characters
            found_match = False
            for j in range(i + 1, len(video_files)):
                video2_path = video_files[j]
                video2_name = os.path.splitext(os.path.basename(video2_path))[0]
                chars2_dict = {char['name']: char for char in video_characters.get(video2_name, [])}
                common_char_names = set(chars1_dict.keys()).intersection(set(chars2_dict.keys()))
                
                # If we found a video with common characters, process this pair
                if common_char_names:
                    pair_name = f"{video1_name} -> {video2_name}"
                    print(f"Found valid pair: {pair_name} (common characters: {list(common_char_names)})")
                    
                    pair_results = {}
                    
                    try:
                        #import cv2
                        # Extract first frames from both videos
                        cap1 = cv2.VideoCapture(video1_path)
                        cap2 = cv2.VideoCapture(video2_path)
                        
                        # Get first frames
                        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        
                        ret1, frame1 = cap1.read()
                        ret2, frame2 = cap2.read()
                        
                        cap1.release()
                        cap2.release()
                        
                        if ret1 and ret2:
                            # Save frames
                            frame1_path = os.path.join(clothes_output_dir, f"{video1_name}_first_frame.jpg")
                            frame2_path = os.path.join(clothes_output_dir, f"{video2_name}_first_frame.jpg")
                            
                            cv2.imwrite(frame1_path, frame1)
                            cv2.imwrite(frame2_path, frame2)
                            
                            # Check clothes consistency for each common character
                            for char_name in common_char_names:
                                try:
                                    # Use the short description as character prompt
                                    char_short = chars1_dict[char_name]['short']
                                    
                                    result = self.clothes_evaluator.check_consistency(
                                        frame1_path, frame2_path, char_short
                                    )
                                    pair_results[char_name] = result
                                    
                                    score = result.get('final_score', 0)
                                    if score is not None:
                                        all_scores.append(score)
                                    
                                    print(f"    {char_name} ({char_short}): score={score}/5")
                                    
                                except Exception as e:
                                    print(f"    {char_name}: Error - {e}")
                                    pair_results[char_name] = {"error": str(e)}
                        
                        else:
                            pair_results["error"] = "Could not extract first frames"
                            print(f"  Error: Could not extract first frames")
                        
                    except Exception as e:
                        pair_results["error"] = str(e)
                        print(f"  Error: {e}")
                    
                    results["video_pairs"][pair_name] = pair_results
                    found_match = True
                    break  # Found the nearest match, stop searching for this video
            
            if not found_match:
                print(f"No subsequent video found with common characters for {video1_name}")
    
        # Calculate summary statistics
        results["summary"] = {
            "average_score": float(np.mean(all_scores)) if all_scores else 0.0,
            "min_score": float(np.min(all_scores)) if all_scores else 0.0,
            "max_score": float(np.max(all_scores)) if all_scores else 0.0,
            "total_comparisons": int(len(all_scores)),
            "valid_pairs_found": len(results["video_pairs"])
        }
        
        print(f"Clothes color consistency summary:")
        print(f"  Valid pairs found: {results['summary']['valid_pairs_found']}")
        print(f"  Avg score: {results['summary']['average_score']:.2f}/5")
        print(f"  Min score: {results['summary']['min_score']:.2f}/5")
        print(f"  Max score: {results['summary']['max_score']:.2f}/5")
        print(f"  Total comparisons: {results['summary']['total_comparisons']}")
        
        return results

    def evaluate_relative_size_consistency(self, video_files: List[str], video_characters: Dict[str, List[Dict[str, str]]], script_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate relative size consistency - for each video, find the nearest subsequent video with >=2 common characters"""
        print("=== Evaluating Relative Size Consistency ===")

        # Ensure evaluator is available
        self._ensure_size_evaluator()
        
        start_time = time.time()
        
        results = {
            "video_pairs": {},
            "summary": {}
        }
        
        consistency_scores = []
        print(f"video_characters: {video_characters}")
        
        # For each video, search forward for the nearest video with >=2 common characters
        for i in range(len(video_files)):
            video1_path = video_files[i]
            video1_name = os.path.splitext(os.path.basename(video1_path))[0]
            chars1 = video_characters.get(video1_name, [])
            chars1_names = {char['name'] for char in chars1}
            
            # Search forward for the nearest video with >=2 common characters
            found_match = False
            for j in range(i + 1, len(video_files)):
                video2_path = video_files[j]
                video2_name = os.path.splitext(os.path.basename(video2_path))[0]
                chars2 = video_characters.get(video2_name, [])
                chars2_names = {char['name'] for char in chars2}
                
                # Find common characters
                common_chars = chars1_names.intersection(chars2_names)
                # print(f"Chars1: {chars1_names}")
                # print(f"Chars2: {chars2_names}")
                # print(f"Common characters between {video1_name} and {video2_name}:")
                # print(common_chars)
                
                # If we found a video with >=2 common characters, process this pair
                if len(common_chars) >= 2:
                    pair_name = f"{video1_name} -> {video2_name}"
                    print(f"Found valid pair: {pair_name} (common characters: {list(common_chars)})")
                    
                    try:
                        # Get character short descriptions for the common characters
                        character_shorts = []
                        for char_name in common_chars:
                            # Find the character short description
                            for char in chars1:
                                if char['name'] == char_name:
                                    character_shorts.append(char['short'])
                                    break
                        
                        # Analyze consistency for this video pair using short descriptions
                        result = self.size_evaluator.analyze_consistency(
                            [video1_path, video2_path], 
                            character_shorts
                        )
                        
                        results["video_pairs"][pair_name] = {
                            "consistency_score": result.consistency_score,
                            "max_variation": result.max_variation,
                            "pair_ratios": result.pair_ratios,
                            "common_characters": list(common_chars)
                        }
                        
                        consistency_scores.append(result.consistency_score)
                        print(f"  Consistency score: {result.consistency_score:.3f}")
                        print(f"  Max variation: {result.max_variation:.3f}")
                        
                    except Exception as e:
                        results["video_pairs"][pair_name] = {
                            "error": str(e),
                            "common_characters": list(common_chars)
                        }
                        print(f"  Error: {e}")
                    
                    found_match = True
                    break  # Found the nearest match, stop searching for this video
            
            if not found_match:
                print(f"No subsequent video found with >=2 common characters for {video1_name}")
        
        end_time = time.time()
        try:
            duration = end_time - start_time
        except NameError:
            print("Warning: start_time was not defined in evaluate_relative_size_consistency, using 0 for duration")
            duration = 0.0
        
        if not consistency_scores:
            print("No valid video pairs found with multiple common characters")
            results["summary"] = {
                "average_consistency_score": 0.0,
                "min_consistency_score": 0.0,
                "max_consistency_score": 0.0,
                "total_comparisons": 0,
                "valid_pairs_found": 0
            }
            print(f"Relative size consistency evaluation completed in {duration:.2f} seconds\n")
            results["duration_seconds"] = duration
            return results
        
        # Calculate summary
        results["summary"] = {
            "average_consistency_score": float(np.mean(consistency_scores)) if consistency_scores else 0.0,
            "min_consistency_score": float(np.min(consistency_scores)) if consistency_scores else 0.0,
            "max_consistency_score": float(np.max(consistency_scores)) if consistency_scores else 0.0,
            "total_comparisons": int(len(consistency_scores)),
            "valid_pairs_found": len(results["video_pairs"])
        }
        
        print(f"Relative size consistency summary:")
        print(f"  Valid pairs found: {results['summary']['valid_pairs_found']}")
        print(f"  Avg consistency score: {results['summary']['average_consistency_score']:.3f}")
        print(f"  Min consistency score: {results['summary']['min_consistency_score']:.3f}")
        print(f"  Max consistency score: {results['summary']['max_consistency_score']:.3f}")
        print(f"Relative size consistency evaluation completed in {duration:.2f} seconds\n")
        
        results["duration_seconds"] = duration
        return results

    def _convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def _contains_resource_exhausted(self, obj: Any) -> bool:
        try:
            if isinstance(obj, dict):
                for v in obj.values():
                    if self._contains_resource_exhausted(v):
                        return True
                return False
            if isinstance(obj, list):
                for item in obj:
                    if self._contains_resource_exhausted(item):
                        return True
                return False
            if isinstance(obj, str):
                s = obj.upper()
                return ("RESOURCE_EXHAUSTED" in s) or (" 429" in s) or (s.startswith("429 "))
            return False
        except Exception:
            return False

    def _deep_merge_dicts(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge two dictionaries without losing existing keys.
        - For dict values: merge recursively.
        - For lists/scalars: incoming overrides base.
        """
        if not isinstance(base, dict):
            return self._convert_numpy_types(incoming)
        merged = dict(base)
        for k, v in (incoming or {}).items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = self._deep_merge_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged

    def save_intermediate_results(self, output_file: str, results: Dict[str, Any]) -> None:
        """Save intermediate results to JSON file"""
        try:
            existing = self.load_existing_results(output_file)
            # Deep-merge existing results with new partial results
            merged = self._deep_merge_dicts(existing, results)
            # Convert all NumPy types to Python native types before saving
            json_safe_results = self._convert_numpy_types(merged)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
            print(f"✓ Intermediate results merged and saved to: {output_file}")
        except Exception as e:
            print(f"⚠️  Error saving intermediate results: {e}")

    def evaluate(self, video_dir: str, script_path: str, characters_dir: str, output_file: str = None, submetrics: Optional[List[str]] = None, existing_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate all video consistency metrics
        
        Args:
            video_dir: Directory containing video files
            script_path: Path to script.json file
            characters_dir: Directory containing character reference images
            output_file: Optional JSON file to save results
            
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Starting Video Consistency Evaluation")
        print(f"Video directory: {video_dir}")
        print(f"Script file: {script_path}")
        print(f"Characters directory: {characters_dir}")
        print(f"Output file: {output_file}")
        print("=" * 70)
        
        total_start_time = time.time()
        
        # Load existing results: prefer provided dict, fallback to file
        if existing_results is not None:
            existing = existing_results
        elif output_file:
            existing = self.load_existing_results(output_file)
        else:
            existing = {}
        
        # Get video files
        video_files = self.get_video_files(video_dir)
        
        # Load script
        script_data = self.load_script(script_path) if script_path and os.path.exists(script_path) else {}
        
        # Get character reference images
        character_refs = self.get_character_ref_images(characters_dir, script_data) if characters_dir and script_data else {}
        print(f"Found references for {len(character_refs)} characters\n")
        
        # Analyze video-character mappings
        video_characters = self.analyze_video_characters(video_files, script_data) if script_data else {}
        print(f"video_characters: {video_characters}")
        
        # Check if video files match existing results
        video_files_match = self.check_video_files_match(existing, video_files)
        if not video_files_match and existing_results:
            print("⚠️  Video files have changed, will re-run all evaluations")
            existing = {}
        
        # Check which evaluations are already complete
        completed_evaluations = []
        evaluation_types = [
            "character_consistency",
            "face_consistency",
            "background_consistency",
            "clothes_color_consistency",
            "relative_size_consistency",
        ]

        # Normalize requested submetrics
        normalized_subs = None
        if submetrics:
            alias = {
                "character": "character_consistency",
                "character_consistency": "character_consistency",
                "face": "face_consistency",
                "face_consistency": "face_consistency",
                "background": "background_consistency",
                "background_consistency": "background_consistency",
                "clothes": "clothes_color_consistency",
                "clothes_color_consistency": "clothes_color_consistency",
                "size": "relative_size_consistency",
                "relative_size_consistency": "relative_size_consistency",
            }
            allowed = set(alias.values())
            normalized_subs = {alias.get(s, s) for s in submetrics if alias.get(s, s) in allowed}
            print(f"Requested submetrics: {', '.join(sorted(normalized_subs))}")

        requested_types = list(normalized_subs) if normalized_subs else evaluation_types

        for eval_type in evaluation_types:
            if eval_type not in requested_types:
                continue
            if self.is_evaluation_complete(existing, eval_type, video_files, video_characters):
                completed_evaluations.append(eval_type)
                print(f"✓ {eval_type} already completed, will reuse existing results")
        
        if completed_evaluations:
            print(f"Found {len(completed_evaluations)} completed evaluations among requested types, will skip them")
        print()
        
        # Create output directory for intermediate results
        if output_file:
            output_dir = os.path.dirname(output_file)
            consistency_output_dir = os.path.join(output_dir, "consistency_analysis")
        else:
            consistency_output_dir = os.path.join(BASE_DIR, "Evaluation", "results", "tmp", "video_consistency")
        
        os.makedirs(consistency_output_dir, exist_ok=True)
        
        # Prepare default placeholders
        default_character = {"summary": {"avg_video_to_video_distance": 0.0, "avg_video_to_reference_distance": 0.0}, "duration_seconds": 0.0, "video_pairs": {}}
        default_face = {"summary": {"avg_video_to_video_distance": 0.0, "avg_video_to_reference_distance": 0.0}, "duration_seconds": 0.0, "video_pairs": {}}
        default_background = {"summary": {"average_distance": 0.0}, "duration_seconds": 0.0, "video_pairs": {}}
        default_clothes = {"summary": {"average_score": 0.0}, "duration_seconds": 0.0, "video_pairs": {}}
        default_size = {"summary": {"average_consistency_score": 0.0}, "duration_seconds": 0.0, "video_pairs": {}}

        # Evaluate consistency metrics (skip completed ones; respect requested types)
        need_run_character = ("character_consistency" in requested_types) and ("character_consistency" not in completed_evaluations) and character_refs
        if need_run_character:
            print("Running character consistency evaluation...")
            character_results = self.evaluate_character_consistency(
                video_files, character_refs, video_characters, consistency_output_dir
            )
            # Save intermediate results after character consistency
            if output_file:
                temp_results = {
                    "video_directory": video_dir,
                    "script_file": script_path,
                    "characters_directory": characters_dir,
                    "total_videos": len(video_files),
                    "video_files": [os.path.basename(f) for f in video_files],
                    "character_references": list(character_refs.keys()),
                    "video_character_mapping": video_characters,
                    "character_consistency": character_results,
                }
                self.save_intermediate_results(output_file, temp_results)
        else:
            if "character_consistency" in existing:
                character_results = existing["character_consistency"]
            else:
                character_results = default_character
        
        need_run_face = ("face_consistency" in requested_types) and ("face_consistency" not in completed_evaluations) and character_refs
        if need_run_face:
            print("Running face consistency evaluation...")
            face_results = self.evaluate_face_consistency(
                video_files, character_refs, video_characters, consistency_output_dir
            )
            # Save intermediate results after face consistency
            if output_file:
                temp_results = {
                    "video_directory": video_dir,
                    "script_file": script_path,
                    "characters_directory": characters_dir,
                    "total_videos": len(video_files),
                    "video_files": [os.path.basename(f) for f in video_files],
                    "character_references": list(character_refs.keys()),
                    "video_character_mapping": video_characters,
                    "character_consistency": character_results,
                    "face_consistency": face_results,
                }
                self.save_intermediate_results(output_file, temp_results)
        else:
            if "face_consistency" in existing:
                face_results = existing["face_consistency"]
            else:
                face_results = default_face
        
        need_run_background = ("background_consistency" in requested_types) and ("background_consistency" not in completed_evaluations)
        if need_run_background:
            print("Running background consistency evaluation...")
            background_results = self.evaluate_background_consistency(
                video_files, consistency_output_dir
            )
            # Save intermediate results after background consistency
            if output_file:
                temp_results = {
                    "video_directory": video_dir,
                    "script_file": script_path,
                    "characters_directory": characters_dir,
                    "total_videos": len(video_files),
                    "video_files": [os.path.basename(f) for f in video_files],
                    "character_references": list(character_refs.keys()),
                    "video_character_mapping": video_characters,
                    "character_consistency": character_results,
                    "face_consistency": face_results,
                    "background_consistency": background_results,
                }
                self.save_intermediate_results(output_file, temp_results)
        else:
            if "background_consistency" in existing:
                background_results = existing["background_consistency"]
            else:
                background_results = default_background
        
        need_run_clothes = ("clothes_color_consistency" in requested_types) and ("clothes_color_consistency" not in completed_evaluations) and video_characters
        if need_run_clothes:
            print("Running clothes color consistency evaluation...")
            clothes_results = self.evaluate_clothes_color_consistency(
                video_files, video_characters, consistency_output_dir
            )
            # Save intermediate results after clothes consistency
            if output_file:
                temp_results = {
                    "video_directory": video_dir,
                    "script_file": script_path,
                    "characters_directory": characters_dir,
                    "total_videos": len(video_files),
                    "video_files": [os.path.basename(f) for f in video_files],
                    "character_references": list(character_refs.keys()),
                    "video_character_mapping": video_characters,
                    "character_consistency": character_results,
                    "face_consistency": face_results,
                    "background_consistency": background_results,
                    "clothes_color_consistency": clothes_results,
                }
                self.save_intermediate_results(output_file, temp_results)
        else:
            if "clothes_color_consistency" in existing:
                clothes_results = existing["clothes_color_consistency"]
            else:
                clothes_results = default_clothes
        
        need_run_size = ("relative_size_consistency" in requested_types) and ("relative_size_consistency" not in completed_evaluations) and video_characters and script_data
        if need_run_size:
            print("Running relative size consistency evaluation...")
            size_results = self.evaluate_relative_size_consistency(
                video_files, video_characters, script_data
            )
            # Save intermediate results after relative size consistency
            if output_file:
                temp_results = {
                    "video_directory": video_dir,
                    "script_file": script_path,
                    "characters_directory": characters_dir,
                    "total_videos": len(video_files),
                    "video_files": [os.path.basename(f) for f in video_files],
                    "character_references": list(character_refs.keys()),
                    "video_character_mapping": video_characters,
                    "character_consistency": character_results,
                    "face_consistency": face_results,
                    "background_consistency": background_results,
                    "clothes_color_consistency": clothes_results,
                    "relative_size_consistency": size_results,
                }
                self.save_intermediate_results(output_file, temp_results)
        else:
            if "relative_size_consistency" in existing:
                size_results = existing["relative_size_consistency"]
            else:
                size_results = default_size
        
        total_end_time = time.time()
        try:
            total_duration = total_end_time - total_start_time
        except NameError:
            print("Warning: total_start_time was not defined, using 0 for total duration")
            total_duration = 0.0
        
        # Include only sub-metrics that are requested or reused
        # Preserve any existing submetrics even if not requested; requested types only control execution
        include_character = need_run_character or ("character_consistency" in existing)
        include_face = need_run_face or ("face_consistency" in existing)
        include_background = need_run_background or ("background_consistency" in existing)
        include_clothes = need_run_clothes or ("clothes_color_consistency" in existing)
        include_size = need_run_size or ("relative_size_consistency" in existing)

        # Recalculate total runtime (count only included sub-metrics)
        total_duration = 0.0
        if include_character:
            total_duration += (character_results.get("duration_seconds") or 0.0)
        if include_face:
            total_duration += (face_results.get("duration_seconds") or 0.0)
        if include_background:
            total_duration += (background_results.get("duration_seconds") or 0.0)
        if include_clothes:
            total_duration += (clothes_results.get("duration_seconds") or 0.0)
        if include_size:
            total_duration += (size_results.get("duration_seconds") or 0.0)

        results = {
            "video_directory": video_dir,
            "script_file": script_path,
            "characters_directory": characters_dir,
            "total_videos": len(video_files),
            "video_files": [os.path.basename(f) for f in video_files],
            "character_references": list(character_refs.keys()),
            "video_character_mapping": video_characters,
        }

        if include_character:
            results["character_consistency"] = character_results
        if include_face:
            results["face_consistency"] = face_results
        if include_background:
            results["background_consistency"] = background_results
        if include_clothes:
            results["clothes_color_consistency"] = clothes_results
        if include_size:
            results["relative_size_consistency"] = size_results

        summary = {}
        if include_character:
            summary["character_v2v_distance"] = (
                (character_results.get("summary", {}) or {}).get("avg_video_to_video_distance", 0.0)
            )
            summary["character_v2r_distance"] = (
                (character_results.get("summary", {}) or {}).get("avg_video_to_reference_distance", 0.0)
            )
        if include_face:
            summary["face_v2v_distance"] = (
                (face_results.get("summary", {}) or {}).get("avg_video_to_video_distance", 0.0)
            )
            summary["face_v2r_distance"] = (
                (face_results.get("summary", {}) or {}).get("avg_video_to_reference_distance", 0.0)
            )
        if include_background:
            summary["background_avg_distance"] = (
                (background_results.get("summary", {}) or {}).get("average_distance", 0.0)
            )
        if include_clothes:
            summary["clothes_avg_score"] = (
                (clothes_results.get("summary", {}) or {}).get("average_score", 0.0)
            )
        if include_size:
            summary["size_avg_consistency"] = (
                (size_results.get("summary", {}) or {}).get("average_consistency_score", 0.0)
            )
        results["summary"] = summary

        timing_info = {"total_duration_seconds": total_duration}
        if include_character:
            timing_info["character_consistency_duration_seconds"] = character_results.get("duration_seconds", 0.0)
        if include_face:
            timing_info["face_consistency_duration_seconds"] = face_results.get("duration_seconds", 0.0)
        if include_background:
            timing_info["background_consistency_duration_seconds"] = background_results.get("duration_seconds", 0.0)
        if include_clothes:
            timing_info["clothes_color_consistency_duration_seconds"] = clothes_results.get("duration_seconds", 0.0)
        if include_size:
            timing_info["relative_size_consistency_duration_seconds"] = size_results.get("duration_seconds", 0.0)
        results["timing_info"] = timing_info
        
        # Print summary
        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Total videos processed: {len(video_files)}")
        if "character_v2v_distance" in results["summary"]:
            print(f"Character consistency (video-to-video):     {results['summary']['character_v2v_distance']:.4f}")
        if "character_v2r_distance" in results["summary"]:
            print(f"Character consistency (video-to-ref):       {results['summary']['character_v2r_distance']:.4f}")
        if "face_v2v_distance" in results["summary"]:
            print(f"Face consistency (video-to-video):          {results['summary']['face_v2v_distance']:.4f}")
        if "face_v2r_distance" in results["summary"]:
            print(f"Face consistency (video-to-ref):            {results['summary']['face_v2r_distance']:.4f}")
        if "background_avg_distance" in results["summary"]:
            print(f"Background consistency:                     {results['summary']['background_avg_distance']:.4f}")
        if "clothes_avg_score" in results["summary"]:
            print(f"Clothes color consistency:                  {results['summary']['clothes_avg_score']:.2f}/5")
        if "size_avg_consistency" in results["summary"]:
            print(f"Relative size consistency:                  {results['summary']['size_avg_consistency']:.3f}")
        print("=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        if "character_consistency_duration_seconds" in results["timing_info"]:
            print(f"Character consistency evaluation time:      {results['timing_info']['character_consistency_duration_seconds']:.2f} seconds")
        if "face_consistency_duration_seconds" in results["timing_info"]:
            print(f"Face consistency evaluation time:           {results['timing_info']['face_consistency_duration_seconds']:.2f} seconds")
        if "background_consistency_duration_seconds" in results["timing_info"]:
            print(f"Background consistency evaluation time:     {results['timing_info']['background_consistency_duration_seconds']:.2f} seconds")
        if "clothes_color_consistency_duration_seconds" in results["timing_info"]:
            print(f"Clothes color consistency evaluation time:  {results['timing_info']['clothes_color_consistency_duration_seconds']:.2f} seconds")
        if "relative_size_consistency_duration_seconds" in results["timing_info"]:
            print(f"Relative size consistency evaluation time:  {results['timing_info']['relative_size_consistency_duration_seconds']:.2f} seconds")
        print(f"Total evaluation time:                      {results['timing_info']['total_duration_seconds']:.2f} seconds")
        
        # Save results if output file specified
        if output_file:
            try:
                # Deep-merge with on-disk existing results to avoid losing prior submetrics
                existing_on_disk = self.load_existing_results(output_file)
                merged_final = self._deep_merge_dicts(existing_on_disk, results)
                # Convert all NumPy types to Python native types before saving
                json_safe_results = self._convert_numpy_types(merged_final)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved (merged) to: {output_file}")

                # Also save a backup with timestamp if any new evaluations were run
                if len(completed_evaluations) < len(requested_types):  # Not all requested evaluations were reused
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = output_file.replace('.json', f'_backup_{timestamp}.json')
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
                    print(f"Backup saved to: {backup_file}")
            except Exception as e:
                print(f"\nError saving results: {e}")
        
        return results


if __name__ == "__main__":
    # Example usage
    evaluator = VideoConsistencyEvaluator(gemini_api_keys=["YOUR_GEMINI_API_KEY"])
    
    # Define paths
    video_dir = "your_video_directory_here"
    script_path = "your_script_json_path_here"
    characters_dir = "your_characters_directory_here"
    output_file = "your_output_json_path_here"
    
    try:
        results = evaluator.evaluate(video_dir, script_path, characters_dir, output_file)
        print("\nEvaluation completed successfully!")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
