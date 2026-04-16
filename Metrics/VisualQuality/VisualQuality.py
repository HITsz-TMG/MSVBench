import os
import json
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

# Note: to reduce initialization overhead, evaluators are lazily loaded on demand


class VisualQualityEvaluator:
    def __init__(self):
        """Initialize with lazy-loaded evaluators"""
        print("Initializing Visual Quality Evaluator (lazy)...")
        # Lazy instantiation: create evaluator only when its metric is first used
        self.dover_evaluator = None
        self.musiq_evaluator = None
        self.bcv_analyzer = None
        self.style_evaluator = None
        print("Evaluators will be loaded on first use.\n")

    def _safe_json_load(self, path: str) -> Dict[str, Any]:
        """Safely load existing JSON file to avoid overwriting previous results"""
        try:
            if path and os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: failed to load existing results from {path}: {e}")
        return {}

    def _deep_merge_dicts(self, base: Any, new: Any) -> Any:
        """Recursively deep-merge dictionaries, preferring values from 'new'.
        - Dict: merge per key; recurse for nested dicts
        - List: prefer 'new' if non-empty, else keep 'base'
        - Other types: return 'new' if not None, else 'base'
        """
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
        """Read old -> deep-merge -> write. Preserves existing sub-metrics without overwriting."""
        if not output_file:
            return
        try:
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            existing = self._safe_json_load(output_file)
            merged = self._deep_merge_dicts(existing, payload)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged, f, indent=2)
            print(f"\nResults merged and saved to: {output_file}")
        except Exception as e:
            print(f"\nError saving results: {e}")

    def _ensure_dover(self):
        if self.dover_evaluator is None:
            from Dover import DOVEREvaluator
            self.dover_evaluator = DOVEREvaluator()
            print("✓ Dover evaluator loaded (lazy)")

    def _ensure_musiq(self):
        if self.musiq_evaluator is None:
            from Musiq import MusiqEvaluator
            self.musiq_evaluator = MusiqEvaluator()
            print("✓ Musiq evaluator loaded (lazy)")

    def _ensure_bcv(self):
        if self.bcv_analyzer is None:
            from BrightnessContrastVariance import VideoQualityAnalyzer
            self.bcv_analyzer = VideoQualityAnalyzer()
            print("✓ Brightness/Contrast/Variance analyzer loaded (lazy)")

    def _ensure_style(self):
        if self.style_evaluator is None:
            from StyleConsistency import StyleConsistencyEvaluator
            self.style_evaluator = StyleConsistencyEvaluator()
            print("✓ Style consistency evaluator loaded (lazy)")

    def get_video_files(self, video_dir: str) -> List[str]:
        """Get all video files from directory in alphabetical order, excluding files with 'complete' in name"""
        if not os.path.exists(video_dir):
            raise ValueError(f"Video directory does not exist: {video_dir}")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
        video_files = []
        
        for file in os.listdir(video_dir):
            if Path(file).suffix.lower() in video_extensions:
                # Filter out files containing 'complete' in their name
                if 'LongLive' in video_dir or 'complete' not in file.lower():
                    video_files.append(os.path.join(video_dir, file))
        
        if not video_files:
            raise ValueError(f"No video files found in directory: {video_dir}")
        
        # Sort alphabetically
        video_files.sort()
        print(f"Found {len(video_files)} video files (excluding files with 'complete' in name):")
        for i, video in enumerate(video_files):
            print(f"  {i+1}. {os.path.basename(video)}")
        print()
        
        return video_files

    def evaluate_dover_scores(self, video_files: List[str]) -> Dict[str, Any]:
        """Evaluate Dover scores for all videos"""
        print("=== Evaluating Dover Scores ===")
        start_time = time.time()
        dover_scores = {}
        # Lazy-load Dover
        self._ensure_dover()
        
        for i, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            print(f"Processing Dover for video {i+1}/{len(video_files)}: {video_name}")
            
            try:
                score = self.dover_evaluator.evaluate(video_path)
                dover_scores[video_name] = score
                print(f"  Dover score: {score:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                dover_scores[video_name] = 0.0
        
        end_time = time.time()
        duration = end_time - start_time
        avg_dover = np.mean(list(dover_scores.values()))
        print(f"Average Dover score: {avg_dover:.4f}")
        print(f"Dover evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "scores": dover_scores,
            "average": avg_dover,
            "duration_seconds": duration
        }

    def evaluate_musiq_scores(self, video_files: List[str]) -> Dict[str, Any]:
        """Evaluate Musiq scores for all videos"""
        print("=== Evaluating Musiq Scores ===")
        start_time = time.time()
        musiq_scores = {}
        # Lazy-load Musiq
        self._ensure_musiq()
        
        for i, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            print(f"Processing Musiq for video {i+1}/{len(video_files)}: {video_name}")
            
            try:
                score = self.musiq_evaluator.evaluate_video(video_path, verbose=False)
                musiq_scores[video_name] = score
                print(f"  Musiq score: {score:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                musiq_scores[video_name] = 0.0
        
        end_time = time.time()
        duration = end_time - start_time
        avg_musiq = np.mean(list(musiq_scores.values()))
        print(f"Average Musiq score: {avg_musiq:.4f}")
        print(f"Musiq evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "scores": musiq_scores,
            "average": avg_musiq,
            "duration_seconds": duration
        }

    def evaluate_bcv_scores(self, video_files: List[str]) -> Dict[str, Any]:
        """Evaluate Brightness/Contrast/Variance scores"""
        print("=== Evaluating Brightness/Contrast/Variance Scores ===")
        start_time = time.time()
        # Lazy-load BCV analyzer
        self._ensure_bcv()
        
        # Internal scores for each video
        internal_scores = {}
        for i, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            print(f"Processing BCV internal for video {i+1}/{len(video_files)}: {video_name}")
            
            try:
                result = self.bcv_analyzer.analyze(video_path)
                if "error" not in result:
                    internal_scores[video_name] = result["score"]
                    print(f"  Internal BCV score: {result['score']:.4f}")
                else:
                    print(f"  Error: {result['error']}")
                    internal_scores[video_name] = 0.0
            except Exception as e:
                print(f"  Error: {e}")
                internal_scores[video_name] = 0.0
        
        # Cross-video scores between adjacent pairs
        cross_scores = {}
        for i in range(len(video_files) - 1):
            video1_name = os.path.basename(video_files[i])
            video2_name = os.path.basename(video_files[i + 1])
            pair_name = f"{video1_name} -> {video2_name}"
            
            print(f"Processing BCV cross-video: {pair_name}")
            
            try:
                result = self.bcv_analyzer.analyze(video_files[i], video_files[i + 1])
                if "error" not in result:
                    cross_scores[pair_name] = result["score"]
                    print(f"  Cross BCV score: {result['score']:.4f}")
                else:
                    print(f"  Error: {result['error']}")
                    cross_scores[pair_name] = 0.0
            except Exception as e:
                print(f"  Error: {e}")
                cross_scores[pair_name] = 0.0
        
        end_time = time.time()
        duration = end_time - start_time
        avg_internal = np.mean(list(internal_scores.values())) if internal_scores else 0.0
        avg_cross = np.mean(list(cross_scores.values())) if cross_scores else 0.0
        
        print(f"Average internal BCV score: {avg_internal:.4f}")
        print(f"Average cross-video BCV score: {avg_cross:.4f}")
        print(f"BCV evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "internal_scores": internal_scores,
            "cross_scores": cross_scores,
            "avg_internal": avg_internal,
            "avg_cross": avg_cross,
            "duration_seconds": duration
        }

    def evaluate_style_consistency_scores(self, video_files: List[str]) -> Dict[str, Any]:
        """Evaluate Style Consistency scores"""
        print("=== Evaluating Style Consistency Scores ===")
        start_time = time.time()
        # Lazy-load style-consistency evaluator
        self._ensure_style()
        
        # Internal scores for each video
        internal_scores = {}
        for i, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            print(f"Processing Style Consistency internal for video {i+1}/{len(video_files)}: {video_name}")
            
            try:
                score = self.style_evaluator.evaluate(video_path)
                internal_scores[video_name] = score
                print(f"  Internal Style Consistency score: {score:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                internal_scores[video_name] = 0.0
        
        # Cross-video scores between adjacent pairs
        cross_scores = {}
        for i in range(len(video_files) - 1):
            video1_name = os.path.basename(video_files[i])
            video2_name = os.path.basename(video_files[i + 1])
            pair_name = f"{video1_name} -> {video2_name}"
            
            print(f"Processing Style Consistency cross-video: {pair_name}")
            
            try:
                score = self.style_evaluator.evaluate(video_files[i], video_files[i + 1])
                cross_scores[pair_name] = score
                print(f"  Cross Style Consistency score: {score:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                cross_scores[pair_name] = 0.0
        
        end_time = time.time()
        duration = end_time - start_time
        avg_internal = np.mean(list(internal_scores.values())) if internal_scores else 0.0
        avg_cross = np.mean(list(cross_scores.values())) if cross_scores else 0.0
        
        print(f"Average internal Style Consistency score: {avg_internal:.4f}")
        print(f"Average cross-video Style Consistency score: {avg_cross:.4f}")
        print(f"Style Consistency evaluation completed in {duration:.2f} seconds\n")
        
        return {
            "internal_scores": internal_scores,
            "cross_scores": cross_scores,
            "avg_internal": avg_internal,
            "avg_cross": avg_cross,
            "duration_seconds": duration
        }

    def evaluate(self, 
                 video_dir: str, 
                 output_file: str = None,
                 submetrics: Optional[List[str]] = None,
                 existing_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate all visual quality metrics for videos in a directory
        
        Args:
            video_dir: Directory containing video files
            output_file: Optional JSON file to save results
            submetrics: Optional. Restrict evaluation to specific sub-metrics; supported names:
              ["dover_scores", "musiq_scores", "brightness_contrast_variance", "style_consistency"]
              Shorthand aliases are also supported:["dover", "musiq", "bcv", "style_consistency"]
            existing_results: Optional existing results dictionary for incremental evaluation and reuse
            
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Starting Visual Quality Evaluation for directory: {video_dir}")
        print("=" * 70)

        total_start_time = time.time()
        
        # Get video files
        video_files = self.get_video_files(video_dir)
        
        # Normalize sub-metric names
        normalized_subs = None
        if submetrics:
            alias = {
                "dover": "dover_scores",
                "dover_scores": "dover_scores",
                "musiq": "musiq_scores",
                "musiq_scores": "musiq_scores",
                "bcv": "brightness_contrast_variance",
                "brightness_contrast_variance": "brightness_contrast_variance",
                "style_consistency": "style_consistency",
            }
            normalized_subs = {alias.get(s, s) for s in submetrics if alias.get(s, s) in {
                "dover_scores", "musiq_scores", "brightness_contrast_variance", "style_consistency"
            }}

        # Read existing module results (passed by orchestrator) for incremental reuse
        existing = existing_results or {}

        # Decide which metrics need to run (if submetrics is not specified, run only missing items)
        need_run_dover = ("dover_scores" not in existing) and ((normalized_subs is None) or ("dover_scores" in normalized_subs))
        need_run_musiq = ("musiq_scores" not in existing) and ((normalized_subs is None) or ("musiq_scores" in normalized_subs))
        need_run_bcv = ("brightness_contrast_variance" not in existing) and ((normalized_subs is None) or ("brightness_contrast_variance" in normalized_subs))
        need_run_style = ("style_consistency" not in existing) and ((normalized_subs is None) or ("style_consistency" in normalized_subs))

        # Evaluate sub-metrics (run missing or explicitly requested items), reuse existing for others
        if need_run_dover:
            dover_results = self.evaluate_dover_scores(video_files)
        else:
            dover_results = {"scores": existing.get("dover_scores", {}), "average": float(np.mean(list(existing.get("dover_scores", {}).values()))) if existing.get("dover_scores") else 0.0, "duration_seconds": (existing.get("timing_info", {}) or {}).get("dover_duration_seconds", 0.0)}

        if need_run_musiq:
            musiq_results = self.evaluate_musiq_scores(video_files)
        else:
            musiq_results = {"scores": existing.get("musiq_scores", {}), "average": float(np.mean(list(existing.get("musiq_scores", {}).values()))) if existing.get("musiq_scores") else 0.0, "duration_seconds": (existing.get("timing_info", {}) or {}).get("musiq_duration_seconds", 0.0)}

        if need_run_bcv:
            bcv_results = self.evaluate_bcv_scores(video_files)
        else:
            bcv_existing = existing.get("brightness_contrast_variance", {})
            bcv_results = {
                "internal_scores": bcv_existing.get("internal_scores", {}),
                "cross_scores": bcv_existing.get("cross_scores", {}),
                "avg_internal": bcv_existing.get("avg_internal", 0.0),
                "avg_cross": bcv_existing.get("avg_cross", 0.0),
                "duration_seconds": (existing.get("timing_info", {}) or {}).get("bcv_duration_seconds", 0.0)
            }

        if need_run_style:
            style_results = self.evaluate_style_consistency_scores(video_files)
        else:
            style_existing = existing.get("style_consistency", {})
            style_results = {
                "internal_scores": style_existing.get("internal_scores", {}),
                "cross_scores": style_existing.get("cross_scores", {}),
                "avg_internal": style_existing.get("avg_internal", 0.0),
                "avg_cross": style_existing.get("avg_cross", 0.0),
                "duration_seconds": (existing.get("timing_info", {}) or {}).get("style_consistency_duration_seconds", 0.0)
            }

        total_end_time = time.time()
        # Total duration is summed from included sub-metrics (reused items use existing timing)
        total_duration = (
            (dover_results.get("duration_seconds") or 0.0) +
            (musiq_results.get("duration_seconds") or 0.0) +
            (bcv_results.get("duration_seconds") or 0.0) +
            (style_results.get("duration_seconds") or 0.0)
        )

        # Include only evaluated or reused sub-metrics
        include_dover = ((normalized_subs is None) or ("dover_scores" in normalized_subs)) and (need_run_dover or ("dover_scores" in existing))
        include_musiq = ((normalized_subs is None) or ("musiq_scores" in normalized_subs)) and (need_run_musiq or ("musiq_scores" in existing))
        include_bcv = ((normalized_subs is None) or ("brightness_contrast_variance" in normalized_subs)) and (need_run_bcv or ("brightness_contrast_variance" in existing))
        include_style = ((normalized_subs is None) or ("style_consistency" in normalized_subs)) and (need_run_style or ("style_consistency" in existing))

        # Recompute total duration (count included sub-metrics only)
        total_duration = 0.0
        if include_dover:
            total_duration += (dover_results.get("duration_seconds") or 0.0)
        if include_musiq:
            total_duration += (musiq_results.get("duration_seconds") or 0.0)
        if include_bcv:
            total_duration += (bcv_results.get("duration_seconds") or 0.0)
        if include_style:
            total_duration += (style_results.get("duration_seconds") or 0.0)

        # Assemble final results conditionally
        results = {
            "video_directory": video_dir,
            "total_videos": len(video_files),
            "video_files": [os.path.basename(f) for f in video_files],
        }

        if include_dover:
            results["dover_scores"] = dover_results["scores"]
        if include_musiq:
            results["musiq_scores"] = musiq_results["scores"]
        if include_bcv:
            results["brightness_contrast_variance"] = {
                "internal_scores": bcv_results["internal_scores"],
                "cross_scores": bcv_results["cross_scores"],
                "avg_internal": bcv_results["avg_internal"],
                "avg_cross": bcv_results["avg_cross"],
            }
        if include_style:
            results["style_consistency"] = {
                "internal_scores": style_results["internal_scores"],
                "cross_scores": style_results["cross_scores"],
                "avg_internal": style_results["avg_internal"],
                "avg_cross": style_results["avg_cross"],
            }

        # Conditional summary
        summary = {}
        if include_dover:
            summary["avg_dover"] = dover_results.get(
                "average",
                float(np.mean(list(dover_results.get("scores", {}).values()))) if dover_results.get("scores") else 0.0,
            )
        if include_musiq:
            summary["avg_musiq"] = musiq_results.get(
                "average",
                float(np.mean(list(musiq_results.get("scores", {}).values()))) if musiq_results.get("scores") else 0.0,
            )
        if include_bcv:
            summary["avg_bcv_internal"] = bcv_results["avg_internal"]
            summary["avg_bcv_cross"] = bcv_results["avg_cross"]
        if include_style:
            summary["avg_style_internal"] = style_results["avg_internal"]
            summary["avg_style_cross"] = style_results["avg_cross"]
        results["summary"] = summary

        # Conditional timing info
        timing_info = {"total_duration_seconds": total_duration}
        if include_dover:
            timing_info["dover_duration_seconds"] = dover_results.get("duration_seconds", 0.0)
        if include_musiq:
            timing_info["musiq_duration_seconds"] = musiq_results.get("duration_seconds", 0.0)
        if include_bcv:
            timing_info["bcv_duration_seconds"] = bcv_results.get("duration_seconds", 0.0)
        if include_style:
            timing_info["style_consistency_duration_seconds"] = style_results.get("duration_seconds", 0.0)
        results["timing_info"] = timing_info
        
        # Print summary
        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Total videos processed: {len(video_files)}")
        if "avg_dover" in results["summary"]:
            print(f"Average Dover score:                    {results['summary']['avg_dover']:.4f}")
        if "avg_musiq" in results["summary"]:
            print(f"Average Musiq score:                    {results['summary']['avg_musiq']:.4f}")
        if "avg_bcv_internal" in results["summary"]:
            print(f"Average BCV internal score:             {results['summary']['avg_bcv_internal']:.4f}")
        if "avg_bcv_cross" in results["summary"]:
            print(f"Average BCV cross-video score:          {results['summary']['avg_bcv_cross']:.4f}")
        if "avg_style_internal" in results["summary"]:
            print(f"Average Style Consistency internal:     {results['summary']['avg_style_internal']:.4f}")
        if "avg_style_cross" in results["summary"]:
            print(f"Average Style Consistency cross-video:  {results['summary']['avg_style_cross']:.4f}")
        print("=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        if "dover_duration_seconds" in results["timing_info"]:
            print(f"Dover evaluation time:                  {results['timing_info']['dover_duration_seconds']:.2f} seconds")
        if "musiq_duration_seconds" in results["timing_info"]:
            print(f"Musiq evaluation time:                  {results['timing_info']['musiq_duration_seconds']:.2f} seconds")
        if "bcv_duration_seconds" in results["timing_info"]:
            print(f"BCV evaluation time:                    {results['timing_info']['bcv_duration_seconds']:.2f} seconds")
        if "style_consistency_duration_seconds" in results["timing_info"]:
            print(f"Style Consistency evaluation time:      {results['timing_info']['style_consistency_duration_seconds']:.2f} seconds")
        print(f"Total evaluation time:                  {results['timing_info']['total_duration_seconds']:.2f} seconds")
        
        # Save results if output file is specified (deep-merge to preserve previous metrics)
        if output_file:
            self._save_merged_results(output_file, results)

        return results


if __name__ == "__main__":
    # Example usage
    evaluator = VisualQualityEvaluator()
    
    # video_dir = "path/to/your/video_dir"  # Replace with an actual video directory.
    # output_file = os.path.join(BASE_DIR, "VisualQuality", "results.json")
    video_dir = "path/to/your/video_dir"  # Replace with an actual video directory.
    output_file = os.path.join(BASE_DIR, "VisualQuality", "msvbench_results.json")
    
    try:
        results = evaluator.evaluate(video_dir, output_file)
        print("\nEvaluation completed successfully!")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
