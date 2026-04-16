#!/usr/bin/env python3
"""
MSVBench - Multi-Story Video Benchmark Evaluation Script

This script integrates all evaluation modules:
- VisualQuality
- StoryVideoAlignment  
- VideoConsistency
- MotionQuality

Usage:
    Run via MSVBench.sh for production evaluation.
    python MSVBench.py
"""

import os
import sys
import json
import traceback
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add module paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(ROOT_DIR, "Metrics")
EVALUATION_DIR = os.path.join(ROOT_DIR, "Evaluation")
DATA_DIR = os.path.join(EVALUATION_DIR, "data")
sys.path.append(os.path.join(METRICS_DIR, "VisualQuality"))
sys.path.append(os.path.join(METRICS_DIR, "StoryVideoAlignment"))
sys.path.append(os.path.join(METRICS_DIR, "VideoConsistency"))
sys.path.append(os.path.join(METRICS_DIR, "MotionQuality"))
sys.path.append(DATA_DIR)

# Lazy import evaluators to avoid unused initialization overhead
import importlib


class MSVBenchEvaluator:
    """Main MSVBench evaluation class that integrates all evaluation modules"""
    
    def __init__(self):
        """Initialize MSVBench evaluator"""
        print("=" * 80)
        print("MSVBench - Multi-Story Video Benchmark Evaluation")
        print("=" * 80)
        
        # Base paths
        self.base_dir = ROOT_DIR
        self.default_input_dir = os.path.join(self.base_dir, "Dataset")
        self.custom_dataset_dir = os.path.join(self.base_dir, "Dataset", "baselineinfo")
        self.video_data_dir = os.path.join(self.base_dir, "Evaluation", "data")
        self.metrics_dir = os.path.join(self.base_dir, "Metrics")
        self.results_dir = os.path.join(self.base_dir, "Evaluation", "results")

        # Gemini API keys from environment only
        self.gemini_api_keys = self._load_gemini_keys_from_env() or []
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize evaluators (lazy loading to save memory)
        self.visual_quality_evaluator = None
        self.story_alignment_evaluator = None
        self.video_consistency_evaluator = None
        self.motion_quality_evaluator = None
        
        print(f"✓ MSVBench evaluator initialized")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Results directory: {self.results_dir}")
        if not self.gemini_api_keys:
            print("  Warning: no Gemini API key detected; Gemini-related metrics may fail.")
        print()

    def _load_gemini_keys_from_env(self) -> Optional[List[str]]:
        """Load Gemini API keys from environment variables.

        Supports:
        - GEMINI_API_KEYS: comma-separated string or JSON array
        - GEMINI_API_KEYS_FILE: a file containing JSON array or newline/comma-separated keys
        Returns None if no valid override found.
        """
        def _normalize(keys: List[str]) -> List[str]:
            cleaned = [str(k).strip() for k in keys if str(k).strip()]
            # dedupe while preserving order
            seen = set()
            result = []
            for k in cleaned:
                if k not in seen:
                    seen.add(k)
                    result.append(k)
            return result

        raw = os.environ.get("GEMINI_API_KEYS")
        file_path = os.environ.get("GEMINI_API_KEYS_FILE")

        # Try inline env first
        if raw:
            # Try JSON array first
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return _normalize(parsed)
            except Exception:
                pass
            # Fallback to CSV
            return _normalize(raw.split(','))

        # Try file-based override
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return _normalize(parsed)
                except Exception:
                    pass
                # Fallback: split by newline or comma
                if '\n' in content:
                    return _normalize(content.splitlines())
                else:
                    return _normalize(content.split(','))
            except Exception:
                return None

        return None

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
                s = str(obj).upper()
                return ("RESOURCE_EXHAUSTED" in s) or (" 429" in s) or (s.startswith("429 "))
            return False
        except Exception:
            return False

    def _filter_submetrics_without_429(self, existing_module: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(existing_module, dict):
            return existing_module
        filtered: Dict[str, Any] = {}
        for k, v in existing_module.items():
            if self._contains_resource_exhausted(v):
                print(f"⚠️  Detected 429 RESOURCE_EXHAUSTED in historical submetric {k}, re-running this submetric")
                continue
            filtered[k] = v
        return filtered

    def get_input_paths(self, story_id: str, method: str) -> Dict[str, str]:
        """
        Get input file paths for a given story ID
        
        Args:
            story_id: Story ID (e.g., "27", "69")
            method: Method name (e.g., "CogVideo", "MMStoryAgent")
            
        Returns:
            Dictionary containing paths to input files
        """
        method_dir = os.path.join(self.custom_dataset_dir, method)
        default_dir = self.default_input_dir
        path_candidates = {
            "prompt_path": [
                os.path.join(method_dir, "prompt", f"{story_id}.txt"),
                os.path.join(default_dir, "prompt", f"{story_id}.txt"),
            ],
            "script_path": [
                os.path.join(method_dir, "script", f"{story_id}.json"),
                os.path.join(default_dir, "script", f"{story_id}.json"),
            ],
            "camera_path": [
                os.path.join(method_dir, "camera", f"{story_id}.txt"),
                os.path.join(default_dir, "camera", f"{story_id}.txt"),
            ],
            "characters_dir": [
                os.path.join(method_dir, "characters", story_id),
                os.path.join(default_dir, "characters", story_id),
            ],
        }

        existing_paths: Dict[str, str] = {}
        for key, candidates in path_candidates.items():
            chosen = next((p for p in candidates if os.path.exists(p)), None)
            if chosen:
                existing_paths[key] = chosen

        return existing_paths

    def get_video_directory(self, method: str, story_id: str) -> str:
        """
        Get video directory path for a given method and story ID
        
        Args:
            method: Method name (e.g., "CogVideo", "Wan2.2-i2v")
            story_id: Story ID (e.g., "27", "69")
            
        Returns:
            Path to video directory
        """
        candidates = [
            os.path.join(self.custom_dataset_dir, method, "videos", story_id),
            os.path.join(self.video_data_dir, "videos", story_id),
        ]
        video_dir = next((p for p in candidates if os.path.exists(p)), None)
        if video_dir is None:
            raise FileNotFoundError(f"Video directory not found. Tried: {candidates}")
        
        # Check if directory contains video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
        video_files = [f for f in os.listdir(video_dir) 
                      if Path(f).suffix.lower() in video_extensions]
        
        if not video_files:
            raise ValueError(f"No video files found in directory: {video_dir}")
        
        return video_dir

    def get_output_path(self, method: str, story_id: str) -> str:
        """
        Get output file path for results
        
        Args:
            method: Method name
            story_id: Story ID
            
        Returns:
            Path to output JSON file
        """
        method_dir = os.path.join(self.results_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        return os.path.join(method_dir, f"{story_id}.json")

    def initialize_visual_quality_evaluator(self):
        """Lazy initialization of Visual Quality evaluator"""
        if self.visual_quality_evaluator is None:
            print("Initializing Visual Quality Evaluator...")
            VisualQualityEvaluator = importlib.import_module('VisualQuality').VisualQualityEvaluator
            self.visual_quality_evaluator = VisualQualityEvaluator()

    def initialize_story_alignment_evaluator(self):
        """Lazy initialization of Story-Video Alignment evaluator"""
        if self.story_alignment_evaluator is None:
            print("Initializing Story-Video Alignment Evaluator...")
            StoryVideoAlignmentEvaluator = importlib.import_module('StoryVideoAlignment').StoryVideoAlignmentEvaluator
            self.story_alignment_evaluator = StoryVideoAlignmentEvaluator(
                gemini_api_keys=self.gemini_api_keys
            )

    def initialize_video_consistency_evaluator(self):
        """Lazy initialization of Video Consistency evaluator"""
        if self.video_consistency_evaluator is None:
            print("Initializing Video Consistency Evaluator...")
            VideoConsistencyEvaluator = importlib.import_module('VideoConsistency').VideoConsistencyEvaluator
            self.video_consistency_evaluator = VideoConsistencyEvaluator(
                gemini_api_keys=self.gemini_api_keys
            )

    def initialize_motion_quality_evaluator(self):
        """Lazy initialization of Motion Quality evaluator"""
        if self.motion_quality_evaluator is None:
            print("Initializing Motion Quality Evaluator...")
            MotionQualityEvaluator = importlib.import_module('MotionQuality').MotionQualityEvaluator
            # MotionQuality lazily accesses Gemini only when related submetrics are triggered
            self.motion_quality_evaluator = MotionQualityEvaluator(
                gemini_api_keys=self.gemini_api_keys
            )

    def evaluate_single_case(self, method: str, story_id: str, 
                           modules: List[str] = None,
                           submetrics: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Evaluate a single method-story combination
        
        Args:
            method: Method name
            story_id: Story ID
            modules: List of modules to evaluate (default: all)
            
        Returns:
            Combined evaluation results
        """
        if modules is None:
            modules = ["visual_quality", "story_alignment", "video_consistency", "motion_quality"]
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {method} - Story {story_id}")
        print(f"Modules: {', '.join(modules)}")
        print(f"{'='*60}")
        
        # Get paths
        try:
            input_paths = self.get_input_paths(story_id, method)
            video_dir = self.get_video_directory(method, story_id)
            output_path = self.get_output_path(method, story_id)
        except (FileNotFoundError, ValueError) as e:
            error_msg = f"Path error for {method}-{story_id}: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg, "method": method, "story_id": story_id}

        print(f"📁 Video directory: {video_dir}")
        print(f"💾 Output path: {output_path}")

        # Load existing combined results if available (for incremental evaluation)
        existing_combined_results: Dict[str, Any] = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_combined_results = json.load(f)
                print(f"🔁 Found existing results, running incremental evaluation: {output_path}")
            except Exception as e:
                print(f"⚠️  Failed to read existing results, recalculating missing items: {e}")
                existing_combined_results = {}
        
        # Combined results
        combined_results = existing_combined_results if existing_combined_results else {}
        if "evaluation_info" not in combined_results:
            combined_results["evaluation_info"] = {}
        combined_results["evaluation_info"].update({
            "method": method,
            "story_id": story_id,
            "video_directory": video_dir,
            "evaluation_timestamp": datetime.now().isoformat(),
            "modules_evaluated": modules
        })
        if "timing_info" not in combined_results:
            combined_results["timing_info"] = {}
        
        # Evaluate each module
        for module in modules:
            print(f"\n🔄 Starting {module} evaluation...")
            module_start_time = time.time()
            
            try:
                if module == "visual_quality":
                    self.initialize_visual_quality_evaluator()
                    existing_module = existing_combined_results.get("visual_quality") if existing_combined_results else None
                    existing_module = self._filter_submetrics_without_429(existing_module)
                    submods = (submetrics or {}).get("visual_quality")
                    if submods:
                        print(f"   Submetrics: {', '.join(submods)}")
                    result = self.visual_quality_evaluator.evaluate(
                        video_dir,
                        output_file=None,
                        submetrics=submods,
                        existing_results=existing_module
                    )
                    combined_results["visual_quality"] = result

                elif module == "story_alignment":
                    self.initialize_story_alignment_evaluator()
                    existing_module = existing_combined_results.get("story_alignment") if existing_combined_results else None
                    existing_module = self._filter_submetrics_without_429(existing_module)
                    submods = (submetrics or {}).get("story_alignment")
                    if submods:
                        print(f"   Submetrics: {', '.join(submods)}")
                    result = self.story_alignment_evaluator.evaluate(
                        video_dir,
                        input_paths.get("prompt_path"),
                        input_paths.get("script_path"),
                        output_file=None,
                        submetrics=submods,
                        existing_results=existing_module
                    )
                    combined_results["story_alignment"] = result

                elif module == "video_consistency":
                    self.initialize_video_consistency_evaluator()
                    existing_module = existing_combined_results.get("video_consistency") if existing_combined_results else None
                    existing_module = self._filter_submetrics_without_429(existing_module)
                    submods = (submetrics or {}).get("video_consistency")
                    if submods:
                        print(f"   Submetrics: {', '.join(submods)}")
                    result = self.video_consistency_evaluator.evaluate(
                        video_dir,
                        input_paths.get("script_path"),
                        input_paths.get("characters_dir"),
                        submetrics=submods,
                        existing_results=existing_module
                    )
                    combined_results["video_consistency"] = result

                elif module == "motion_quality":
                    self.initialize_motion_quality_evaluator()
                    existing_module = existing_combined_results.get("motion_quality") if existing_combined_results else None
                    existing_module = self._filter_submetrics_without_429(existing_module)
                    submods = (submetrics or {}).get("motion_quality")
                    if submods:
                        print(f"   Submetrics: {', '.join(submods)}")
                    result = self.motion_quality_evaluator.evaluate(
                        video_dir,
                        input_paths.get("prompt_path"),
                        input_paths.get("camera_path"),
                        output_file=None,
                        submetrics=submods,
                        existing_results=existing_module
                    )
                    combined_results["motion_quality"] = result
                
                module_end_time = time.time()
                module_duration = module_end_time - module_start_time
                combined_results["timing_info"][f"{module}_duration_seconds"] = module_duration
                
                print(f"✅ {module} evaluation completed successfully in {module_duration:.2f} seconds")
                
            except Exception as e:
                module_end_time = time.time()
                module_duration = module_end_time - module_start_time
                combined_results["timing_info"][f"{module}_duration_seconds"] = module_duration
                
                error_msg = f"Error in {module} evaluation: {str(e)}"
                print(f"❌ {error_msg}")
                print(f"Traceback: {traceback.format_exc()}")
                combined_results[module] = {
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                }
        
        # Save results
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Results saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
        
        return combined_results

    def evaluate_batch(self, story_ids: List[str], methods: List[str], 
                      modules: List[str] = None,
                      submetrics: Optional[Dict[str, List[str]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple method-story combinations
        
        Args:
            story_ids: List of story IDs
            methods: List of method names
            modules: List of modules to evaluate (default: all)
            
        Returns:
            Dictionary of all evaluation results
        """
        print(f"\n🚀 Starting batch evaluation")
        print(f"📊 Story IDs: {story_ids}")
        print(f"🔧 Methods: {methods}")
        print(f"📦 Modules: {modules or 'all'}")
        if submetrics:
            try:
                pretty = {k: ",".join(v) for k, v in submetrics.items()}
                print(f"🧩 Submetrics: {pretty}")
            except Exception:
                print(f"🧩 Submetrics provided")
        
        all_results = {}
        total_cases = len(story_ids) * len(methods)
        current_case = 0
        
        for method in methods:
            method_results = {}
            
            for story_id in story_ids:
                current_case += 1
                print(f"\n📈 Progress: {current_case}/{total_cases}")
                
                case_key = f"{method}_{story_id}"
                try:
                    result = self.evaluate_single_case(method, story_id, modules, submetrics=submetrics)
                    method_results[story_id] = result
                except Exception as e:
                    error_msg = f"Failed to evaluate {method}-{story_id}: {e}"
                    print(f"❌ {error_msg}")
                    method_results[story_id] = {
                        "error": error_msg,
                        "method": method,
                        "story_id": story_id
                    }
            
            all_results[method] = method_results
        
        print(f"\n🎉 Batch evaluation completed!")
        print(f"📊 Total cases processed: {total_cases}")
        
        return all_results


def main():
    """Entry point for single-case evaluation, designed for MSVBench.sh."""
    try:
        method = os.environ.get("METHOD")
        story_id = os.environ.get("STORY_ID")
        if not method or not story_id:
            print("Usage from shell: METHOD=<method> STORY_ID=<id> [MODULES=...] [SUBMETRICS=...] python MSVBench.py")
            print("Recommended: run via `sh MSVBench.sh`.")
            sys.exit(1)

        modules_env = os.environ.get("MODULES", "").strip()
        modules = [m.strip() for m in modules_env.split(',') if m.strip()] if modules_env else None

        submetrics_env = os.environ.get("SUBMETRICS", "").strip()
        submetrics: Optional[Dict[str, List[str]]] = None
        if submetrics_env:
            submetrics = {}
            for item in submetrics_env.split(';'):
                item = item.strip()
                if not item or '=' not in item:
                    continue
                mod, subs = item.split('=', 1)
                mod = mod.strip()
                subs_list = [s.strip() for s in subs.split(',') if s.strip()]
                if mod and subs_list:
                    submetrics[mod] = subs_list

        print("Evaluation config:")
        print(f"  Method: {method}")
        print(f"  Story ID: {story_id}")
        print(f"  Modules: {modules or 'all'}")
        if submetrics:
            print(f"  Submetrics: {submetrics}")
        print()

        evaluator = MSVBenchEvaluator()
        _ = evaluator.evaluate_single_case(method, story_id, modules=modules, submetrics=submetrics)
        print("\n✅ Evaluation finished.")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
