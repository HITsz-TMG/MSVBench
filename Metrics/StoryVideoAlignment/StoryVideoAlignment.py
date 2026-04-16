import os
import json
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional
import traceback

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
TOOLS_DIR = os.path.join(BASE_DIR, "Tools")

"""
Lazy loading within module: instantiate each component only when its sub-metric is evaluated.
"""

# Delay-import components to avoid unnecessary initialization and resource use
VideoVQAScore = None
VideoDetectionEvaluator = None
ShotPerspectiveAligner = None
VideoChangeAnalyzer = None
VideoPromptConsistency = None

import sys
sys.path.append(TOOLS_DIR)
from gemini_api import GeminiAPI


class StoryVideoAlignmentEvaluator:
    def __init__(self, gemini_api_keys=None):
        """Initialize with lazy-loaded alignment evaluators"""
        print("Initializing Story-Video Alignment Evaluator (lazy)...")
        # Keys and proxy
        self.gemini_api_keys = gemini_api_keys
        self.proxy = "YOUR_PROXY_URL"  # Set to None if no proxy is needed
        # Lazy components
        self.vqa_evaluator = None
        self.gemini_api = None
        self.shot_perspective_evaluator = None
        self.state_persistence_analyzer = None
        print("Components will be loaded on first use.\n")

    def _safe_json_load(self, path: str) -> Dict[str, Any]:
        try:
            if path and os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: failed to load existing results from {path}: {e}")
        return {}

    def _deep_merge_dicts(self, base: Any, new: Any) -> Any:
        if isinstance(base, dict) and isinstance(new, dict):
            merged = dict(base)
            for k, v in new.items():
                if k in merged:
                    merged[k] = self._deep_merge_dicts(merged[k], v)
                else:
                    merged[k] = v
            return merged
        if isinstance(base, list) and isinstance(new, list):
            return new if len(new) > 0 else base
        return new if (new is not None) else base

    def _save_merged_results(self, output_file: str, payload: Dict[str, Any]) -> None:
        if not output_file:
            return
        try:
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            existing = self._safe_json_load(output_file)
            merged = self._deep_merge_dicts(existing, payload)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            print(f"\nResults merged and saved to: {output_file}")
        except Exception as e:
            print(f"\nError saving results: {e}")

    def _ensure_vqa(self):
        global VideoVQAScore
        if self.vqa_evaluator is None:
            if VideoVQAScore is None:
                from VQAScore import VideoVQAScore as _VideoVQAScore
                VideoVQAScore = _VideoVQAScore
            self.vqa_evaluator = VideoVQAScore()
            print("✓ VQA Score evaluator loaded (lazy)")

    def _ensure_gemini(self):
        if self.gemini_api is None:
            self.gemini_api = GeminiAPI(
                api_keys=self.gemini_api_keys,
                proxy=self.proxy
            )
            print("✓ Gemini API initialized (lazy)")

    def _ensure_shot(self):
        global ShotPerspectiveAligner
        if self.shot_perspective_evaluator is None:
            self._ensure_gemini()
            if ShotPerspectiveAligner is None:
                from ShotPerspectiveAlignment import ShotPerspectiveAligner as _ShotPerspectiveAligner
                ShotPerspectiveAligner = _ShotPerspectiveAligner
            self.shot_perspective_evaluator = ShotPerspectiveAligner(self.gemini_api)
            print("✓ Shot Perspective Alignment evaluator loaded (lazy)")

    def _ensure_state(self):
        global VideoChangeAnalyzer
        if self.state_persistence_analyzer is None:
            if VideoChangeAnalyzer is None:
                from StateShiftPersistence import VideoChangeAnalyzer as _VideoChangeAnalyzer
                VideoChangeAnalyzer = _VideoChangeAnalyzer
            self.state_persistence_analyzer = VideoChangeAnalyzer(
                api_keys=self.gemini_api_keys,
                proxy=self.proxy
            )
            print("✓ State Shift Persistence analyzer loaded (lazy)")

    def _ensure_consistency_class(self):
        global VideoPromptConsistency
        if VideoPromptConsistency is None:
            from StoryVideoConsistency import VideoPromptConsistency as _VideoPromptConsistency
            VideoPromptConsistency = _VideoPromptConsistency

    def _ensure_detection(self):
        global VideoDetectionEvaluator
        if VideoDetectionEvaluator is None:
            from DetectionCountScore import VideoDetectionEvaluator as _VideoDetectionEvaluator
            VideoDetectionEvaluator = _VideoDetectionEvaluator
            print("✓ Detection Count Evaluator class loaded (lazy)")

    def _convert_numpy_types(self, obj):
        """Convert NumPy types to native Python types for JSON serialization"""
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

    def load_prompt_file(self, prompt_path: str) -> List[str]:
        """Load prompts from text file, one per line"""
        if not os.path.exists(prompt_path):
            raise ValueError(f"Prompt file does not exist: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Loaded {len(prompts)} prompts from {prompt_path}")
        return prompts

    def load_script_file(self, script_path: str) -> Dict:
        """Load script data from JSON file"""
        if not os.path.exists(script_path):
            raise ValueError(f"Script file does not exist: {script_path}")
        
        with open(script_path, 'r', encoding='utf-8') as f:
            script_data = json.load(f)
        
        print(f"Loaded script data from {script_path}")
        return script_data

    def evaluate_vqa_scores(self, video_files: List[str], prompts: List[str]) -> Dict[str, Any]:
        """Evaluate VQA scores for all video-prompt pairs"""
        print("=== Evaluating VQA Scores ===")
        start_time = time.time()
        vqa_scores = {}
        # Lazy-load VQA evaluator
        self._ensure_vqa()
        
        # Ensure we have the same number of videos and prompts
        min_length = min(len(video_files), len(prompts))
        if len(video_files) != len(prompts):
            print(f"Warning: {len(video_files)} videos but {len(prompts)} prompts. Using first {min_length} of each.")
        
        for i in range(min_length):
            video_path = video_files[i]
            prompt = prompts[i]
            video_name = os.path.basename(video_path)
            
            print(f"Processing VQA for video {i+1}/{min_length}: {video_name}")
            
            try:
                score = self.vqa_evaluator.calculate_single_video_score(video_path, prompt)
                if score is not None:
                    vqa_scores[video_name] = score
                    print(f"  VQA score: {score:.4f}")
                else:
                    print(f"  Failed to calculate VQA score")
                    vqa_scores[video_name] = 0.0
            except Exception as e:
                print(f"  Error: {e}")
                vqa_scores[video_name] = 0.0
        
        end_time = time.time()
        duration = end_time - start_time
        avg_vqa = float(np.mean(list(vqa_scores.values()))) if vqa_scores else 0.0
        print(f"Average VQA score: {avg_vqa:.4f}")
        print(f"VQA evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "scores": vqa_scores,
            "average": avg_vqa,
            "duration_seconds": duration
        }

    def evaluate_detection_scores(self, video_files: List[str], prompts: List[str], script_data: Dict) -> Dict[str, Any]:
        """Evaluate Detection Count scores for all videos"""
        print("=== Evaluating Detection Count Scores ===")
        start_time = time.time()
        detection_scores = {}
        
        # Extract character information from script
        characters_json = json.dumps(script_data.get("characters", []), ensure_ascii=False)
        
        # Lazy-load Detection evaluator class
        self._ensure_detection()
        
        # Ensure we have the same number of videos and prompts
        min_length = min(len(video_files), len(prompts))
        
        for i in range(min_length):
            video_path = video_files[i]
            prompt = prompts[i]
            video_name = os.path.basename(video_path)
            
            print(f"Processing Detection for video {i+1}/{min_length}: {video_name}")
            
            try:
                evaluator = VideoDetectionEvaluator(characters_json, prompt, video_path)
                score = evaluator.evaluate()
                detection_scores[video_name] = score
                print(f"  Detection score: {score:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                detection_scores[video_name] = 0.0
        
        end_time = time.time()
        duration = end_time - start_time
        avg_detection = float(np.mean(list(detection_scores.values()))) if detection_scores else 0.0
        print(f"Average Detection score: {avg_detection:.4f}")
        print(f"Detection evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "scores": detection_scores,
            "average": avg_detection,
            "duration_seconds": duration
        }

    def evaluate_shot_perspective_alignment(self, video_dir: str, script_data: Dict) -> Dict[str, Any]:
        """Evaluate Shot Perspective Alignment for the video directory"""
        print("=== Evaluating Shot Perspective Alignment ===")
        start_time = time.time()
        # Lazy-load ShotPerspectiveAligner and GeminiAPI
        self._ensure_shot()
        
        try:
            results = self.shot_perspective_evaluator.process_script_and_video(script_data, video_dir)
            
            alignment_results = results.get('alignment_results', {})
            avg_distance = alignment_results.get('average_distance_score', 0.0)
            avg_angle = alignment_results.get('average_angle_score', 0.0)
            avg_combined = alignment_results.get('average_combined_score', 0.0)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Average distance alignment score: {avg_distance:.4f}")
            print(f"Average angle alignment score: {avg_angle:.4f}")
            print(f"Average combined alignment score: {avg_combined:.4f}")
            print(f"Shot Perspective Alignment evaluation completed in {duration:.2f} seconds\n")
            
            return {
                "distance_score": avg_distance,
                "angle_score": avg_angle,
                "combined_score": avg_combined,
                "detailed_results": alignment_results,
                "success": True,
                "duration_seconds": duration
            }
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Error in shot perspective alignment: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Shot Perspective Alignment evaluation failed in {duration:.2f} seconds\n")
            return {
                "distance_score": 0.0,
                "angle_score": 0.0,
                "combined_score": 0.0,
                "detailed_results": {},
                "success": False,
                "error": str(e),
                "duration_seconds": duration
            }

    def evaluate_state_shift_persistence(self, video_files: List[str], script_data: Dict) -> Dict[str, Any]:
        """Evaluate State Shift Persistence for the complete video"""
        print("=== Evaluating State Shift Persistence ===")
        start_time = time.time()
        # Lazy-load state persistence analyzer
        self._ensure_state()
        
        try:
            # For state persistence, we need the complete video
            # Check if there's a complete movie file, otherwise use the first video
            complete_video_path = None
            
            # Look for complete movie file
            video_dir = os.path.dirname(video_files[0])
            for potential_complete in [
                os.path.join(video_dir, "complete_movie.mp4"),
            ]:
                if os.path.exists(potential_complete):
                    complete_video_path = potential_complete
                    break
            
            if complete_video_path is None:
                # Create complete_movie.mp4 by concatenating videos in alphabetical order
                complete_video_path = os.path.join(video_dir, "complete_movie.mp4")
                print(f"No complete movie found, creating by concatenating {len(video_files)} videos...")
                
                # Create a temporary file list for ffmpeg
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    for video_file in video_files:
                        f.write(f"file '{video_file}'\n")
                    temp_list_file = f.name
                
                try:
                    # Use ffmpeg to concatenate videos
                    import subprocess
                    cmd = [
                        'ffmpeg', '-f', 'concat', '-safe', '0', 
                        '-i', temp_list_file, 
                        '-c', 'copy', 
                        complete_video_path, 
                        '-y'  # Overwrite output file if exists
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"Successfully created complete movie: {os.path.basename(complete_video_path)}")
                    else:
                        print(f"Error creating complete movie: {result.stderr}")
                        # Fallback to using the longest video file
                        complete_video_path = max(video_files, key=lambda x: os.path.getsize(x))
                        print(f"Using largest video as fallback: {os.path.basename(complete_video_path)}")
                        
                except Exception as e:
                    print(f"Error during video concatenation: {e}")
                    # Fallback to using the longest video file
                    complete_video_path = max(video_files, key=lambda x: os.path.getsize(x))
                    print(f"Using largest video as fallback: {os.path.basename(complete_video_path)}")
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_list_file)
                    except:
                        pass
            else:
                print(f"Using existing complete movie: {os.path.basename(complete_video_path)}")
            
            # Run full analysis
            results = self.state_persistence_analyzer.full_analysis(complete_video_path, script_data)
            
            final_result = results.get('final_result', {})
            final_score = final_result.get('final_score', 0.0)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"State Shift Persistence score: {final_score:.4f}")
            print(f"State Shift Persistence evaluation completed in {duration:.2f} seconds\n")
            
            return {
                "persistence_score": final_score,
                "detailed_results": final_result,
                "analysis_results": results,
                "success": True,
                "duration_seconds": duration
            }
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Error in state shift persistence: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"State Shift Persistence evaluation failed in {duration:.2f} seconds\n")
            return {
                "persistence_score": 0.0,
                "detailed_results": {},
                "analysis_results": {},
                "success": False,
                "error": str(e),
                "duration_seconds": duration
            }

    def evaluate_story_video_consistency(self, video_files: List[str], prompts: List[str]) -> Dict[str, Any]:
        """Evaluate Story-Video Consistency for all video-prompt pairs"""
        print("=== Evaluating Story-Video Consistency ===")
        start_time = time.time()
        consistency_scores = {}
        # Lazy-load VideoPromptConsistency class
        self._ensure_consistency_class()
        
        # Ensure we have the same number of videos and prompts
        min_length = min(len(video_files), len(prompts))
        
        for i in range(min_length):
            video_path = video_files[i]
            prompt = prompts[i]
            video_name = os.path.basename(video_path)
            
            print(f"Processing Consistency for video {i+1}/{min_length}: {video_name}")
            
            # try:
            consistency_checker = VideoPromptConsistency(video_path, prompt)
            similarity_score, _ = consistency_checker.calculate_consistency(verbose=False)
            consistency_scores[video_name] = similarity_score
            print(f"  Consistency score: {similarity_score:.4f}")
            # except Exception as e:
            #     print(f"  Error: {e}")
            #     consistency_scores[video_name] = 0.0
        
        end_time = time.time()
        duration = end_time - start_time
        avg_consistency = float(np.mean(list(consistency_scores.values()))) if consistency_scores else 0.0
        print(f"Average Story-Video Consistency score: {avg_consistency:.4f}")
        print(f"Story-Video Consistency evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "scores": consistency_scores,
            "average": avg_consistency,
            "duration_seconds": duration
        }

    def evaluate(self, 
                 video_dir: str, 
                 prompt_path: str, 
                 script_path: str, 
                 output_file: str = None,
                 submetrics: Optional[List[str]] = None,
                 existing_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate all story-video alignment metrics
        
        Args:
            video_dir: Directory containing video files
            prompt_path: Path to prompt text file
            script_path: Path to script JSON file
            output_file: Optional JSON file to save results
            submetrics: Optional. Evaluate only specified sub-metrics; supports names or aliases:
              ["vqa_scores"/"vqa", "detection_scores"/"detection", "shot_perspective_alignment"/"shot_perspective",
               "state_shift_persistence"/"state_persistence", "story_video_consistency"/"consistency"]
            existing_results: Optional existing module-results dictionary for incremental evaluation and reuse
            
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Starting Story-Video Alignment Evaluation")
        print(f"Video directory: {video_dir}")
        print(f"Prompt file: {prompt_path}")
        print(f"Script file: {script_path}")
        print("=" * 70)
        
        total_start_time = time.time()
        
        # Load input data
        video_files = self.get_video_files(video_dir)
        prompts = self.load_prompt_file(prompt_path) if prompt_path and os.path.exists(prompt_path) else []
        script_data = self.load_script_file(script_path) if script_path and os.path.exists(script_path) else {}
        
        # Normalize sub-metric names
        normalized_subs = None
        if submetrics:
            alias = {
                "vqa_scores": "vqa_scores",
                "vqa": "vqa_scores",
                "detection_scores": "detection_scores",
                "detection": "detection_scores",
                "shot_perspective_alignment": "shot_perspective_alignment",
                "shot_perspective": "shot_perspective_alignment",
                "state_shift_persistence": "state_shift_persistence",
                "state_persistence": "state_shift_persistence",
                "story_video_consistency": "story_video_consistency",
                "consistency": "story_video_consistency",
            }
            allowed = set(alias.values())
            normalized_subs = {alias.get(s, s) for s in submetrics if alias.get(s, s) in allowed}

        existing = existing_results or {}

        # Decide sub-metrics to run (if submetrics is not specified, run only missing items)
        print(f"Submetrics: {normalized_subs}")
        def _should_run(key: str) -> bool:
            if (normalized_subs is not None) and (key not in normalized_subs):
                return False
            val = existing.get(key)
            if not val:
                return True
            if self._contains_resource_exhausted(val):
                print(f"Detected 429 RESOURCE_EXHAUSTED in historical {key}; re-evaluating")
                return True
            return False

        need_run_vqa = _should_run("vqa_scores")
        need_run_detection = _should_run("detection_scores")
        need_run_shot = _should_run("shot_perspective_alignment")
        need_run_state = _should_run("state_shift_persistence")
        need_run_consistency = _should_run("story_video_consistency")
        print(f"Need to run VQA: {need_run_vqa}")
        print(f"Need to run Detection: {need_run_detection}")
        print(f"Need to run Shot Perspective Alignment: {need_run_shot}")
        print(f"Need to run State Shift Persistence: {need_run_state}")
        print(f"Need to run Story-Video Consistency: {need_run_consistency}")

        # Evaluate or reuse each sub-metric
        if need_run_vqa and prompts:
            vqa_results = self.evaluate_vqa_scores(video_files, prompts)
        else:
            vqa_scores = existing.get("vqa_scores", {})
            vqa_avg = (existing.get("summary", {}) or {}).get("raw_avg_vqa_score", float(np.mean(list(vqa_scores.values()))) if vqa_scores else 0.0)
            vqa_duration = (existing.get("timing_info", {}) or {}).get("vqa_duration_seconds", 0.0)
            vqa_results = {"scores": vqa_scores, "average": vqa_avg, "duration_seconds": vqa_duration}

        if need_run_detection and prompts and script_data:
            detection_results = self.evaluate_detection_scores(video_files, prompts, script_data)
        else:
            det_scores = existing.get("detection_scores", {})
            det_avg = (existing.get("summary", {}) or {}).get("raw_avg_detection_score", float(np.mean(list(det_scores.values()))) if det_scores else 0.0)
            det_duration = (existing.get("timing_info", {}) or {}).get("detection_duration_seconds", 0.0)
            detection_results = {"scores": det_scores, "average": det_avg, "duration_seconds": det_duration}

        if need_run_shot and script_data:
            shot_alignment = self.evaluate_shot_perspective_alignment(video_dir, script_data)
        else:
            shot_alignment = existing.get("shot_perspective_alignment", {"distance_score": 0.0, "angle_score": 0.0, "combined_score": 0.0, "detailed_results": {}, "success": False, "duration_seconds": (existing.get("timing_info", {}) or {}).get("shot_perspective_duration_seconds", 0.0)})

        if need_run_state and script_data:
            state_persistence = self.evaluate_state_shift_persistence(video_files, script_data)
        else:
            state_persistence = existing.get("state_shift_persistence", {"persistence_score": 0.0, "detailed_results": {}, "analysis_results": {}, "success": False, "duration_seconds": (existing.get("timing_info", {}) or {}).get("state_persistence_duration_seconds", 0.0)})

        if need_run_consistency and prompts:
            consistency_results = self.evaluate_story_video_consistency(video_files, prompts)
        else:
            cons_scores = existing.get("story_video_consistency", {})
            cons_avg = (existing.get("summary", {}) or {}).get("raw_avg_consistency_score", float(np.mean(list(cons_scores.values()))) if cons_scores else 0.0)
            cons_duration = (existing.get("timing_info", {}) or {}).get("story_video_consistency_duration_seconds", 0.0)
            consistency_results = {"scores": cons_scores, "average": cons_avg, "duration_seconds": cons_duration}
        
        total_end_time = time.time()
        # Unified total duration as sum of sub-metric durations (reuse existing timing where applicable)
        total_duration = (
            (vqa_results.get("duration_seconds") or 0.0) +
            (detection_results.get("duration_seconds") or 0.0) +
            (shot_alignment.get("duration_seconds") or 0.0) +
            (state_persistence.get("duration_seconds") or 0.0) +
            (consistency_results.get("duration_seconds") or 0.0)
        )
        
        # Calculate raw scores
        raw_avg_vqa = vqa_results.get("average", 0.0)
        raw_avg_detection = detection_results.get("average", 0.0)
        raw_shot_perspective = float(shot_alignment.get("combined_score", 0.0))
        raw_state_persistence = float(state_persistence.get("persistence_score", 0.0))
        raw_avg_consistency = consistency_results.get("average", 0.0)
        
        # Normalize scores to 0-100 scale
        # VQA, Detection, Consistency: 0-1 -> 0-100
        normalized_vqa = raw_avg_vqa * 100.0
        normalized_detection = raw_avg_detection * 100.0
        normalized_consistency = raw_avg_consistency * 100.0
        
        # Shot Perspective, State Persistence: 0-5 -> 0-100
        normalized_shot_perspective = (raw_shot_perspective / 5.0) * 100.0
        normalized_state_persistence = (raw_state_persistence / 5.0) * 100.0
        
        # Include only evaluated or reused sub-metrics
        include_vqa = ((normalized_subs is None) or ("vqa_scores" in normalized_subs)) and (need_run_vqa or ("vqa_scores" in existing))
        include_detection = ((normalized_subs is None) or ("detection_scores" in normalized_subs)) and (need_run_detection or ("detection_scores" in existing))
        include_shot = ((normalized_subs is None) or ("shot_perspective_alignment" in normalized_subs)) and (need_run_shot or ("shot_perspective_alignment" in existing))
        include_state = ((normalized_subs is None) or ("state_shift_persistence" in normalized_subs)) and (need_run_state or ("state_shift_persistence" in existing))
        include_consistency = ((normalized_subs is None) or ("story_video_consistency" in normalized_subs)) and (need_run_consistency or ("story_video_consistency" in existing))

        # Recompute total duration
        total_duration = 0.0
        if include_vqa:
            total_duration += (vqa_results.get("duration_seconds") or 0.0)
        if include_detection:
            total_duration += (detection_results.get("duration_seconds") or 0.0)
        if include_shot:
            total_duration += (shot_alignment.get("duration_seconds") or 0.0)
        if include_state:
            total_duration += (state_persistence.get("duration_seconds") or 0.0)
        if include_consistency:
            total_duration += (consistency_results.get("duration_seconds") or 0.0)

        results = {
            "evaluation_info": {
                "video_directory": video_dir,
                "prompt_file": prompt_path,
                "script_file": script_path,
                "total_videos": len(video_files),
                "total_prompts": len(prompts),
                "video_files": [os.path.basename(f) for f in video_files]
            }
        }

        if include_vqa:
            results["vqa_scores"] = vqa_results.get("scores", {})
        if include_detection:
            results["detection_scores"] = detection_results.get("scores", {})
        if include_shot:
            results["shot_perspective_alignment"] = {
                "distance_score": shot_alignment.get("distance_score", 0.0),
                "angle_score": shot_alignment.get("angle_score", 0.0),
                "combined_score": shot_alignment.get("combined_score", 0.0),
                "detailed_results": shot_alignment.get("detailed_results", {}),
                "success": shot_alignment.get("success", False)
            }
        if include_state:
            results["state_shift_persistence"] = {
                "persistence_score": state_persistence.get("persistence_score", 0.0),
                "detailed_results": state_persistence.get("detailed_results", {}),
                "analysis_results": state_persistence.get("analysis_results", {}),
                "success": state_persistence.get("success", False)
            }
        if include_consistency:
            results["story_video_consistency"] = consistency_results.get("scores", {})

        summary = {}
        if include_vqa:
            summary["raw_avg_vqa_score"] = raw_avg_vqa
            summary["avg_vqa_score"] = normalized_vqa
        if include_detection:
            summary["raw_avg_detection_score"] = raw_avg_detection
            summary["avg_detection_score"] = normalized_detection
        if include_shot:
            summary["raw_shot_perspective_combined_score"] = raw_shot_perspective
            summary["shot_perspective_combined_score"] = normalized_shot_perspective
        if include_state:
            summary["raw_state_persistence_score"] = raw_state_persistence
            summary["state_persistence_score"] = normalized_state_persistence
        if include_consistency:
            summary["raw_avg_consistency_score"] = raw_avg_consistency
            summary["avg_consistency_score"] = normalized_consistency
        results["summary"] = summary

        timing_info = {"total_duration_seconds": total_duration}
        if include_vqa:
            timing_info["vqa_duration_seconds"] = vqa_results.get("duration_seconds", 0.0)
        if include_detection:
            timing_info["detection_duration_seconds"] = detection_results.get("duration_seconds", 0.0)
        if include_shot:
            timing_info["shot_perspective_duration_seconds"] = shot_alignment.get("duration_seconds", 0.0)
        if include_state:
            timing_info["state_persistence_duration_seconds"] = state_persistence.get("duration_seconds", 0.0)
        if include_consistency:
            timing_info["story_video_consistency_duration_seconds"] = consistency_results.get("duration_seconds", 0.0)
        results["timing_info"] = timing_info
        
        # Calculate overall alignment score (weighted average) using normalized scores
        weights = {
            "vqa": 0.25,
            "detection": 0.20,
            "shot_perspective": 0.20,
            "state_persistence": 0.15,
            "consistency": 0.20
        }
        
        overall_score = float(
            normalized_vqa * weights["vqa"] +
            normalized_detection * weights["detection"] +
            normalized_shot_perspective * weights["shot_perspective"] +
            normalized_state_persistence * weights["state_persistence"] +
            normalized_consistency * weights["consistency"]
        )
        
        results["summary"]["overall_alignment_score"] = overall_score
        results["summary"]["scoring_weights"] = weights
        
        # Convert all NumPy types to native Python types for JSON serialization
        results = self._convert_numpy_types(results)
        
        # Print summary
        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Total videos processed: {len(video_files)}")
        if "avg_vqa_score" in results["summary"]:
            print(f"Average VQA score:                      {results['summary']['avg_vqa_score']:.2f}/100")
        if "avg_detection_score" in results["summary"]:
            print(f"Average Detection score:                {results['summary']['avg_detection_score']:.2f}/100")
        if "shot_perspective_combined_score" in results["summary"]:
            print(f"Shot Perspective combined score:        {results['summary']['shot_perspective_combined_score']:.2f}/100")
        if "state_persistence_score" in results["summary"]:
            print(f"State Shift Persistence score:          {results['summary']['state_persistence_score']:.2f}/100")
        if "avg_consistency_score" in results["summary"]:
            print(f"Average Story-Video Consistency score:  {results['summary']['avg_consistency_score']:.2f}/100")
        print(f"Overall Alignment Score:                {overall_score:.2f}/100")
        print()
        print("Raw scores (original scale):")
        if "raw_avg_vqa_score" in results["summary"]:
            print(f"Raw VQA score:                          {results['summary']['raw_avg_vqa_score']:.4f}")
        if "raw_avg_detection_score" in results["summary"]:
            print(f"Raw Detection score:                    {results['summary']['raw_avg_detection_score']:.4f}")
        if "raw_shot_perspective_combined_score" in results["summary"]:
            print(f"Raw Shot Perspective score:             {results['summary']['raw_shot_perspective_combined_score']:.4f}/5")
        if "raw_state_persistence_score" in results["summary"]:
            print(f"Raw State Persistence score:            {results['summary']['raw_state_persistence_score']:.4f}/5")
        if "raw_avg_consistency_score" in results["summary"]:
            print(f"Raw Consistency score:                  {results['summary']['raw_avg_consistency_score']:.4f}")
        print("=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        if "vqa_duration_seconds" in results["timing_info"]:
            print(f"VQA evaluation time:                    {results['timing_info']['vqa_duration_seconds']:.2f} seconds")
        if "detection_duration_seconds" in results["timing_info"]:
            print(f"Detection evaluation time:              {results['timing_info']['detection_duration_seconds']:.2f} seconds")
        if "shot_perspective_duration_seconds" in results["timing_info"]:
            print(f"Shot Perspective evaluation time:       {results['timing_info']['shot_perspective_duration_seconds']:.2f} seconds")
        if "state_persistence_duration_seconds" in results["timing_info"]:
            print(f"State Persistence evaluation time:      {results['timing_info']['state_persistence_duration_seconds']:.2f} seconds")
        if "story_video_consistency_duration_seconds" in results["timing_info"]:
            print(f"Story-Video Consistency evaluation time: {results['timing_info']['story_video_consistency_duration_seconds']:.2f} seconds")
        print(f"Total evaluation time:                  {results['timing_info']['total_duration_seconds']:.2f} seconds")
        
        # Save results if output file specified (deep-merge to preserve previous metrics)
        if output_file:
            self._save_merged_results(output_file, results)
        
        return results


if __name__ == "__main__":
    # Example usage
    evaluator = StoryVideoAlignmentEvaluator()
    
    # Define paths
    # video_dir = "path/to/your/video_dir"  # Replace with an actual video directory.
    # prompt_path = "path/to/your/prompt.txt"  # Replace with an actual prompt file.
    # script_path = "path/to/your/script.json"  # Replace with an actual script file.
    # output_file = os.path.join(BASE_DIR, "StoryVideoAlignment", "eval_res", "results.json")
    video_dir = "path/to/your/video_dir"  # Replace with an actual video directory.
    prompt_path = "path/to/your/prompt.txt"  # Replace with an actual prompt file.
    script_path = "path/to/your/script.json"  # Replace with an actual script file.
    output_file = os.path.join(BASE_DIR, "StoryVideoAlignment", "eval_res", "msvbench_results.json")
    
    try:
        results = evaluator.evaluate(video_dir, prompt_path, script_path, output_file)
        print("\nStory-Video Alignment Evaluation completed successfully!")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
