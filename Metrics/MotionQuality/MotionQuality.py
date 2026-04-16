import os
import json
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional
import importlib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MotionQualityEvaluator:
    def __init__(self, gemini_api_keys=None):
        """Initialize evaluator with lazy-loaded components"""
        print("Initializing Motion Quality Evaluator (lazy)...")

        self.gemini_api_keys = gemini_api_keys
        self.gemini_proxy = "YOUR_PROXY_URL"

        # Lazy-loaded components
        self.gemini_api = None
        self.action_recognition = None
        self.action_strength = None
        self.camera_control = None
        self.physical_plausibility = None
        self.physical_interaction = None

        print("Motion Quality Evaluator initialized with lazy components.\n")

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
                json.dump(merged, f, indent=2, default=str)
            print(f"\nResults merged and saved to: {output_file}")
        except Exception as e:
            print(f"\nError saving results: {e}")

    def _ensure_gemini_api(self):
        if self.gemini_api is None:
            try:
                GeminiAPI = importlib.import_module('Tools.gemini_api').GeminiAPI
            except Exception:
                # Fallback path import when module path differs
                module = importlib.import_module('gemini_api')
                GeminiAPI = getattr(module, 'GeminiAPI')
            self.gemini_api = GeminiAPI(api_keys=self.gemini_api_keys, proxy=self.gemini_proxy)

    def _ensure_action_recognition(self):
        if self.action_recognition is None:
            ActionRecognition = importlib.import_module('ActionRecognition').ActionRecognition
            self.action_recognition = ActionRecognition(gemini_proxy=self.gemini_proxy)

    def _ensure_action_strength(self):
        if self.action_strength is None:
            ActionStrengthEvaluator = importlib.import_module('ActionStrength').ActionStrengthEvaluator
            self.action_strength = ActionStrengthEvaluator()

    def _ensure_camera_control(self):
        if self.camera_control is None:
            CameraControl = importlib.import_module('CameraControl').CameraControl
            self.camera_control = CameraControl()

    def _ensure_physical_plausibility(self):
        if self.physical_plausibility is None:
            self._ensure_gemini_api()
            PhysicalPlausibility = importlib.import_module('PhysicalPlausibility').PhysicalPlausibility
            self.physical_plausibility = PhysicalPlausibility(self.gemini_api)

    def _ensure_physical_interaction(self):
        if self.physical_interaction is None:
            self._ensure_gemini_api()
            PhysicalInteractionAccuracy = importlib.import_module('PhysicalInteractionAccuracy').PhysicalInteractionAccuracy
            self.physical_interaction = PhysicalInteractionAccuracy(self.gemini_api)

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

    def read_prompts(self, prompt_file: str) -> List[str]:
        """Read prompts from file, one per line"""
        if not os.path.exists(prompt_file):
            raise ValueError(f"Prompt file does not exist: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Read {len(prompts)} prompts from {prompt_file}")
        return prompts

    def read_camera_instructions(self, camera_file: str) -> List[str]:
        """Read camera instructions from file, one per line"""
        if not os.path.exists(camera_file):
            raise ValueError(f"Camera instruction file does not exist: {camera_file}")
        
        with open(camera_file, 'r', encoding='utf-8') as f:
            instructions = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Read {len(instructions)} camera instructions from {camera_file}")
        return instructions

    def evaluate_action_recognition(self, video_files: List[str], prompts: List[str]) -> Dict[str, Any]:
        """Evaluate Action Recognition scores for all videos"""
        print("=== Evaluating Action Recognition Scores ===")
        start_time = time.time()
        self._ensure_action_recognition()
        
        action_scores = {}
        detailed_results = {}
        
        for i, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            prompt = prompts[i] if i < len(prompts) else ""
            
            print(f"Processing Action Recognition for video {i+1}/{len(video_files)}: {video_name}")
            print(f"  Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"  Prompt: {prompt}")
            
            try:
                results = self.action_recognition.evaluate_video_with_prompt(
                    video_path, prompt, return_predictions=True
                )
                
                # Extract score based on analysis method
                if results['analysis_method'] == 'character_specific':
                    score = results['scores']['overall_action_score']
                else:
                    score = results['scores'].get('action_score', 0.0)
                
                action_scores[video_name] = score
                detailed_results[video_name] = results
                print(f"  Action Recognition score: {score:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                action_scores[video_name] = 0.0
                detailed_results[video_name] = {"error": str(e)}
        
        # Calculate average score excluding videos with errors
        valid_scores = []
        for video_name, detailed_result in detailed_results.items():
            # Check if this video has an error
            if "error" not in detailed_result:
                valid_scores.append(action_scores[video_name])
        
        if valid_scores:
            avg_action = np.mean(valid_scores)
            print(f"Average Action Recognition score: {avg_action:.4f} (calculated from {len(valid_scores)}/{len(action_scores)} valid videos)")
        else:
            avg_action = 0.0
            print(f"Average Action Recognition score: {avg_action:.4f} (no valid videos found)")
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Action Recognition evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "scores": action_scores,
            "detailed_results": detailed_results,
            "average": avg_action,
            "duration_seconds": duration
        }

    def evaluate_action_strength(self, video_files: List[str]) -> Dict[str, Any]:
        """Evaluate Action Strength scores for all videos"""
        print("=== Evaluating Action Strength Scores ===")
        start_time = time.time()
        self._ensure_action_strength()
        
        strength_scores = {}
        
        for i, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            print(f"Processing Action Strength for video {i+1}/{len(video_files)}: {video_name}")
            
            try:
                score = self.action_strength.calculate_flow_score(video_path)
                strength_scores[video_name] = score
                print(f"  Action Strength score: {score:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                strength_scores[video_name] = 0.0
        
        avg_strength = np.mean(list(strength_scores.values()))
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Average Action Strength score: {avg_strength:.4f}")
        print(f"Action Strength evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "scores": strength_scores,
            "average": avg_strength,
            "duration_seconds": duration
        }

    def evaluate_camera_control(self, video_files: List[str], camera_instructions: List[str]) -> Dict[str, Any]:
        """Evaluate Camera Control scores for all videos"""
        print("=== Evaluating Camera Control Scores ===")
        start_time = time.time()
        self._ensure_camera_control()
        
        camera_scores = {}
        detailed_results = {}
        
        for i, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            instruction = camera_instructions[i] if i < len(camera_instructions) else None
            
            print(f"Processing Camera Control for video {i+1}/{len(video_files)}: {video_name}")
            if instruction:
                print(f"  Camera instruction: {instruction}")
            else:
                print("  No camera instruction provided")
            
            try:
                scene, outfile, imgs, analysis = self.camera_control.process_video(
                    video_path, instruction
                )
                
                score = analysis['score'] if analysis and analysis['score'] is not None else 0.0
                camera_scores[video_name] = score
                detailed_results[video_name] = analysis
                
                print(f"  Camera Control score: {score}/5")
                if analysis and analysis['score_explanation']:
                    print(f"  Explanation: {analysis['score_explanation'][:100]}...")
                
            except Exception as e:
                print(f"  Error: {e}")
                camera_scores[video_name] = 0.0
                detailed_results[video_name] = {"error": str(e)}
        
        # Filter out scores of 0 when calculating average
        non_zero_scores = [score for score in camera_scores.values() if score > 0]
        if non_zero_scores:
            avg_camera = np.mean(non_zero_scores)
            print(f"Average Camera Control score: {avg_camera:.4f} (calculated from {len(non_zero_scores)} non-zero scores out of {len(camera_scores)} total)")
        else:
            avg_camera = 0.0
            print(f"Average Camera Control score: {avg_camera:.4f} (all scores are 0)")
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Camera Control evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "scores": camera_scores,
            "detailed_results": detailed_results,
            "average": avg_camera,
            "duration_seconds": duration
        }

    def evaluate_physical_plausibility(self, video_files: List[str], prompts: List[str]) -> Dict[str, Any]:
        """Evaluate Physical Plausibility scores for all videos"""
        print("=== Evaluating Physical Plausibility Scores ===")
        start_time = time.time()
        self._ensure_physical_plausibility()
        
        plausibility_scores = {}
        detailed_results = {}
        valid_scores = []  # Store valid scores only (exclude character_count=0 cases)
        
        for i, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            prompt = prompts[i] if i < len(prompts) else ""
            
            print(f"Processing Physical Plausibility for video {i+1}/{len(video_files)}: {video_name}")
            print(f"  Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"  Prompt: {prompt}")
            
            try:
                results = self.physical_plausibility.score_physical_plausibility(video_path, prompt)
                score = results['final_score']
                
                plausibility_scores[video_name] = score
                detailed_results[video_name] = results
                
                # Check whether this is a character_count=0 case
                expected_characters = results.get('expected_characters', {})
                character_count = expected_characters.get('character_count', 0)
                
                if character_count > 0:
                    valid_scores.append(score)
                    print(f"  Physical Plausibility score: {score:.4f}/5")
                else:
                    print(f"  Physical Plausibility score: {score:.4f}/5 (excluded from average - no moving characters expected)")
                
            except Exception as e:
                print(f"  Error: {e}")
                plausibility_scores[video_name] = 0.0
                detailed_results[video_name] = {"error": str(e)}
        
        # Calculate average score using valid scores only (exclude character_count=0 cases)
        if valid_scores:
            avg_plausibility = np.mean(valid_scores)
            print(f"Average Physical Plausibility score: {avg_plausibility:.4f} (based on {len(valid_scores)} videos with moving characters)")
        else:
            avg_plausibility = 0.0
            print(f"Average Physical Plausibility score: {avg_plausibility:.4f} (no videos with moving characters found)")
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Physical Plausibility evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "scores": plausibility_scores,
            "detailed_results": detailed_results,
            "average": avg_plausibility,
            "valid_count": len(valid_scores),  # Number of valid videos
            "total_count": len(video_files),    # Total number of videos
            "duration_seconds": duration
        }

    def evaluate_physical_interaction_accuracy(self, video_files: List[str], prompts: List[str]) -> Dict[str, Any]:
        """Evaluate Physical Interaction Accuracy scores for all videos"""
        print("=== Evaluating Physical Interaction Accuracy Scores ===")
        start_time = time.time()
        self._ensure_physical_interaction()
        
        interaction_scores = {}
        detailed_results = {}
        
        for i, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            prompt = prompts[i] if i < len(prompts) else ""
            
            print(f"Processing Physical Interaction Accuracy for video {i+1}/{len(video_files)}: {video_name}")
            print(f"  Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"  Prompt: {prompt}")
            
            try:
                results = self.physical_interaction.score_physical_interaction_accuracy(video_path, prompt)
                score = results['final_score']
                
                interaction_scores[video_name] = score
                detailed_results[video_name] = results
                print(f"  Physical Interaction Accuracy score: {score:.4f}/5")
                
            except Exception as e:
                print(f"  Error: {e}")
                interaction_scores[video_name] = 0.0
                detailed_results[video_name] = {"error": str(e)}
        
        avg_interaction = np.mean(list(interaction_scores.values()))
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Average Physical Interaction Accuracy score: {avg_interaction:.4f}")
        print(f"Physical Interaction Accuracy evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "scores": interaction_scores,
            "detailed_results": detailed_results,
            "average": avg_interaction,
            "duration_seconds": duration
        }

    def evaluate(self, 
                 video_dir: str, 
                 prompt_file: str = None, 
                 camera_file: str = None, 
                 output_file: str = None,
                 submetrics: Optional[List[str]] = None,
                 existing_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate all motion quality metrics for videos in a directory
        
        Args:
            video_dir: Directory containing video files
            prompt_file: File containing prompts (one per line)
            camera_file: File containing camera instructions (one per line)
            output_file: Optional JSON file to save results
            
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Starting Motion Quality Evaluation for directory: {video_dir}")
        print("=" * 80)
        
        total_start_time = time.time()
        
        # Get video files and read instruction files
        video_files = self.get_video_files(video_dir)
        prompts = self.read_prompts(prompt_file) if prompt_file and os.path.exists(prompt_file) else []
        camera_instructions = self.read_camera_instructions(camera_file) if camera_file and os.path.exists(camera_file) else []
        
        # Validate that we have matching numbers
        if prompts and camera_instructions:
            min_count = min(len(video_files), len(prompts), len(camera_instructions))
            if len(video_files) != len(prompts) or len(video_files) != len(camera_instructions):
                print(f"Warning: Mismatch in counts - Videos: {len(video_files)}, Prompts: {len(prompts)}, Camera: {len(camera_instructions)}")
                print(f"Will process first {min_count} items")
                video_files = video_files[:min_count]
                prompts = prompts[:min_count]
                camera_instructions = camera_instructions[:min_count]
        elif prompts:
            min_count = min(len(video_files), len(prompts))
            if len(video_files) != len(prompts):
                print(f"Warning: Mismatch in counts - Videos: {len(video_files)}, Prompts: {len(prompts)}")
                print(f"Will process first {min_count} items")
                video_files = video_files[:min_count]
                prompts = prompts[:min_count]
        elif camera_instructions:
            min_count = min(len(video_files), len(camera_instructions))
            if len(video_files) != len(camera_instructions):
                print(f"Warning: Mismatch in counts - Videos: {len(video_files)}, Camera: {len(camera_instructions)}")
                print(f"Will process first {min_count} items")
                video_files = video_files[:min_count]
                camera_instructions = camera_instructions[:min_count]

        
        # Normalize submetrics
        normalized_subs = None
        if submetrics:
            alias = {
                "action": "action_recognition",
                "action_recognition": "action_recognition",
                "strength": "action_strength",
                "action_strength": "action_strength",
                "camera": "camera_control",
                "camera_control": "camera_control",
                "plausibility": "physical_plausibility",
                "physical_plausibility": "physical_plausibility",
                "interaction": "physical_interaction_accuracy",
                "physical_interaction_accuracy": "physical_interaction_accuracy",
            }
            allowed = set(alias.values())
            normalized_subs = {alias.get(s, s) for s in submetrics if alias.get(s, s) in allowed}

        existing = existing_results or {}

        def _should_run(key: str, require_prompts: bool = False, require_camera: bool = False) -> bool:
            if (normalized_subs is not None) and (key not in normalized_subs):
                return False
            if require_prompts and not prompts:
                return False
            if require_camera and not camera_instructions:
                return False
            val = existing.get(key)
            if not val:
                return True
            if self._contains_resource_exhausted(val):
                print(f"Detected 429 RESOURCE_EXHAUSTED in historical {key} results, re-running evaluation")
                return True
            return False

        need_run_action = _should_run("action_recognition")
        need_run_strength = _should_run("action_strength")
        need_run_camera = _should_run("camera_control", require_camera=True)
        need_run_plausibility = _should_run("physical_plausibility", require_prompts=True)
        need_run_interaction = _should_run("physical_interaction_accuracy", require_prompts=True)

        # Evaluate or reuse each submetric
        if need_run_action:
            action_recognition_results = self.evaluate_action_recognition(video_files, prompts)
        else:
            action_recognition_results = existing.get("action_recognition", {"scores": {}, "average": 0.0, "duration_seconds": (existing.get("timing_info", {}) or {}).get("action_recognition_duration_seconds", 0.0)})

        if need_run_strength:
            action_strength_results = self.evaluate_action_strength(video_files)
        else:
            action_strength_results = existing.get("action_strength", {"scores": {}, "average": 0.0, "duration_seconds": (existing.get("timing_info", {}) or {}).get("action_strength_duration_seconds", 0.0)})

        if need_run_camera:
            camera_control_results = self.evaluate_camera_control(video_files, camera_instructions)
        else:
            camera_control_results = existing.get("camera_control", {"scores": {}, "average": 0.0, "duration_seconds": (existing.get("timing_info", {}) or {}).get("camera_control_duration_seconds", 0.0)})

        if need_run_plausibility:
            physical_plausibility_results = self.evaluate_physical_plausibility(video_files, prompts)
        else:
            pp = existing.get("physical_plausibility", {})
            physical_plausibility_results = {
                "scores": pp.get("scores", {}),
                "detailed_results": pp.get("detailed_results", {}),
                "average": pp.get("average", 0.0),
                "valid_count": pp.get("valid_count", 0),
                "total_count": pp.get("total_count", 0),
                "duration_seconds": (existing.get("timing_info", {}) or {}).get("physical_plausibility_duration_seconds", 0.0)
            }

        if need_run_interaction:
            physical_interaction_results = self.evaluate_physical_interaction_accuracy(video_files, prompts)
        else:
            pia = existing.get("physical_interaction_accuracy", {})
            physical_interaction_results = {
                "scores": pia.get("scores", {}),
                "detailed_results": pia.get("detailed_results", {}),
                "average": pia.get("average", 0.0),
                "duration_seconds": (existing.get("timing_info", {}) or {}).get("physical_interaction_accuracy_duration_seconds", 0.0)
            }
        
        total_end_time = time.time()
        total_duration = (
            (action_recognition_results.get("duration_seconds") or 0.0) +
            (action_strength_results.get("duration_seconds") or 0.0) +
            (camera_control_results.get("duration_seconds") or 0.0) +
            (physical_plausibility_results.get("duration_seconds") or 0.0) +
            (physical_interaction_results.get("duration_seconds") or 0.0)
        )
        
        # Calculate raw scores
        raw_action_recognition = action_recognition_results.get("average", float(np.mean(list(action_recognition_results.get("scores", {}).values()))) if action_recognition_results.get("scores") else 0.0)
        raw_action_strength = action_strength_results.get("average", float(np.mean(list(action_strength_results.get("scores", {}).values()))) if action_strength_results.get("scores") else 0.0)
        raw_camera_control = camera_control_results.get("average", float(np.mean(list(camera_control_results.get("scores", {}).values()))) if camera_control_results.get("scores") else 0.0)
        raw_physical_plausibility = physical_plausibility_results.get("average", 0.0)
        raw_physical_interaction = physical_interaction_results.get("average", 0.0)
        
        # Normalize scores to 0-100 scale
        # Action Recognition: 0-1 -> 0-100 (multiply by 100)
        normalized_action_recognition = raw_action_recognition * 100.0
        
        # Other scores: 0-5 -> 0-100 (multiply by 20)
        normalized_action_strength = raw_action_strength * 20.0
        normalized_camera_control = raw_camera_control * 20.0
        normalized_physical_plausibility = raw_physical_plausibility * 20.0
        normalized_physical_interaction = raw_physical_interaction * 20.0
        
        # Include only submetrics and summaries that are evaluated or reused
        include_action = ((normalized_subs is None) or ("action_recognition" in normalized_subs)) and (need_run_action or ("action_recognition" in existing))
        include_strength = ((normalized_subs is None) or ("action_strength" in normalized_subs)) and (need_run_strength or ("action_strength" in existing))
        include_camera = ((normalized_subs is None) or ("camera_control" in normalized_subs)) and (need_run_camera or ("camera_control" in existing))
        include_plausibility = ((normalized_subs is None) or ("physical_plausibility" in normalized_subs)) and (need_run_plausibility or ("physical_plausibility" in existing))
        include_interaction = ((normalized_subs is None) or ("physical_interaction_accuracy" in normalized_subs)) and (need_run_interaction or ("physical_interaction_accuracy" in existing))

        # Recalculate total runtime
        total_duration = 0.0
        if include_action:
            total_duration += (action_recognition_results.get("duration_seconds") or 0.0)
        if include_strength:
            total_duration += (action_strength_results.get("duration_seconds") or 0.0)
        if include_camera:
            total_duration += (camera_control_results.get("duration_seconds") or 0.0)
        if include_plausibility:
            total_duration += (physical_plausibility_results.get("duration_seconds") or 0.0)
        if include_interaction:
            total_duration += (physical_interaction_results.get("duration_seconds") or 0.0)

        results = {
            "evaluation_info": {
                "video_directory": video_dir,
                "prompt_file": prompt_file,
                "camera_file": camera_file,
                "total_videos": len(video_files),
                "video_files": [os.path.basename(f) for f in video_files]
            },
        }

        if include_action:
            results["action_recognition"] = action_recognition_results
        if include_strength:
            results["action_strength"] = action_strength_results
        if include_camera:
            results["camera_control"] = camera_control_results
        if include_plausibility:
            results["physical_plausibility"] = physical_plausibility_results
        if include_interaction:
            results["physical_interaction_accuracy"] = physical_interaction_results

        summary = {}
        if include_action:
            summary["raw_avg_action_recognition"] = raw_action_recognition
            summary["avg_action_recognition"] = normalized_action_recognition
        if include_strength:
            summary["raw_avg_action_strength"] = raw_action_strength
            summary["avg_action_strength"] = normalized_action_strength
        if include_camera:
            summary["raw_avg_camera_control"] = raw_camera_control
            summary["avg_camera_control"] = normalized_camera_control
        if include_plausibility:
            summary["raw_avg_physical_plausibility"] = raw_physical_plausibility
            summary["avg_physical_plausibility"] = normalized_physical_plausibility
        if include_interaction:
            summary["raw_avg_physical_interaction"] = raw_physical_interaction
            summary["avg_physical_interaction"] = normalized_physical_interaction
        results["summary"] = summary

        timing_info = {"total_duration_seconds": total_duration}
        if include_action:
            timing_info["action_recognition_duration_seconds"] = action_recognition_results.get("duration_seconds", 0.0)
        if include_strength:
            timing_info["action_strength_duration_seconds"] = action_strength_results.get("duration_seconds", 0.0)
        if include_camera:
            timing_info["camera_control_duration_seconds"] = camera_control_results.get("duration_seconds", 0.0)
        if include_plausibility:
            timing_info["physical_plausibility_duration_seconds"] = physical_plausibility_results.get("duration_seconds", 0.0)
        if include_interaction:
            timing_info["physical_interaction_accuracy_duration_seconds"] = physical_interaction_results.get("duration_seconds", 0.0)
        results["timing_info"] = timing_info
        
        # Calculate overall motion quality score (weighted average) using normalized scores
        weights = {
            "action_recognition": 0.25,
            "action_strength": 0.15,
            "camera_control": 0.20,
            "physical_plausibility": 0.20,
            "physical_interaction": 0.20
        }
        
        overall_score = 0.0
        total_weight = 0.0

        if include_action:
            overall_score += normalized_action_recognition * weights["action_recognition"]
            total_weight += weights["action_recognition"]
        if include_strength:
            overall_score += normalized_action_strength * weights["action_strength"]
            total_weight += weights["action_strength"]
        if include_camera:
            overall_score += normalized_camera_control * weights["camera_control"]
            total_weight += weights["camera_control"]
        if include_plausibility:
            overall_score += normalized_physical_plausibility * weights["physical_plausibility"]
            total_weight += weights["physical_plausibility"]
        if include_interaction:
            overall_score += normalized_physical_interaction * weights["physical_interaction"]
            total_weight += weights["physical_interaction"]

        if total_weight > 0:
            overall_score /= total_weight
        
        results["summary"]["overall_motion_quality_score"] = overall_score
        results["summary"]["scoring_weights"] = weights
        
        # Print summary
        print("=" * 80)
        print("FINAL MOTION QUALITY SUMMARY")
        print("=" * 80)
        print(f"Total videos processed: {len(video_files)}")
        if "avg_action_recognition" in results["summary"]:
            print(f"Average Action Recognition score:       {results['summary']['avg_action_recognition']:.2f}/100")
        if "avg_action_strength" in results["summary"]:
            print(f"Average Action Strength score:          {results['summary']['avg_action_strength']:.2f}/100")
        if "avg_camera_control" in results["summary"]:
            print(f"Average Camera Control score:           {results['summary']['avg_camera_control']:.2f}/100")
        if "avg_physical_plausibility" in results["summary"]:
            print(f"Average Physical Plausibility score:    {results['summary']['avg_physical_plausibility']:.2f}/100")
        if "avg_physical_interaction" in results["summary"]:
            print(f"Average Physical Interaction score:     {results['summary']['avg_physical_interaction']:.2f}/100")
        print(f"Overall Motion Quality Score:           {results['summary']['overall_motion_quality_score']:.2f}/100")
        print()
        print("Raw scores (original scale):")
        if "raw_avg_action_recognition" in results["summary"]:
            print(f"Raw Action Recognition score:           {results['summary']['raw_avg_action_recognition']:.4f}")
        if "raw_avg_action_strength" in results["summary"]:
            print(f"Raw Action Strength score:              {results['summary']['raw_avg_action_strength']:.4f}")
        if "raw_avg_camera_control" in results["summary"]:
            print(f"Raw Camera Control score:               {results['summary']['raw_avg_camera_control']:.4f}/5")
        if "raw_avg_physical_plausibility" in results["summary"]:
            print(f"Raw Physical Plausibility score:        {results['summary']['raw_avg_physical_plausibility']:.4f}/5")
        if "raw_avg_physical_interaction" in results["summary"]:
            print(f"Raw Physical Interaction score:         {results['summary']['raw_avg_physical_interaction']:.4f}/5")
        print("=" * 80)
        print("TIMING SUMMARY")
        print("=" * 80)
        if "action_recognition_duration_seconds" in results["timing_info"]:
            print(f"Action Recognition evaluation time:      {results['timing_info']['action_recognition_duration_seconds']:.2f} seconds")
        if "action_strength_duration_seconds" in results["timing_info"]:
            print(f"Action Strength evaluation time:         {results['timing_info']['action_strength_duration_seconds']:.2f} seconds")
        if "camera_control_duration_seconds" in results["timing_info"]:
            print(f"Camera Control evaluation time:          {results['timing_info']['camera_control_duration_seconds']:.2f} seconds")
        if "physical_plausibility_duration_seconds" in results["timing_info"]:
            print(f"Physical Plausibility evaluation time:   {results['timing_info']['physical_plausibility_duration_seconds']:.2f} seconds")
        if "physical_interaction_accuracy_duration_seconds" in results["timing_info"]:
            print(f"Physical Interaction evaluation time:    {results['timing_info']['physical_interaction_accuracy_duration_seconds']:.2f} seconds")
        print(f"Total evaluation time:                   {results['timing_info']['total_duration_seconds']:.2f} seconds")
        
        # Save results if output file specified (deep-merge to preserve previous metrics)
        if output_file:
            self._save_merged_results(output_file, results)
        
        return results


if __name__ == "__main__":
    # Example usage
    evaluator = MotionQualityEvaluator(gemini_api_keys=["YOUR_GEMINI_API_KEY"])
    
    video_dir = "your_video_directory_here"
    prompt_file = "your_prompt_file_path_here"
    camera_file = "your_camera_file_path_here"
    output_file = "your_output_file_path_here"
    
    try:
        results = evaluator.evaluate(video_dir, prompt_file, camera_file, output_file)
        print("\nMotion Quality Evaluation completed successfully!")
    except Exception as e:
        print(f"\nMotion Quality Evaluation failed: {e}")
