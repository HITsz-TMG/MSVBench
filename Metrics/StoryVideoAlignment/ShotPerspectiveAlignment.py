import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.gemini_api import GeminiAPI
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

class ShotPerspectiveAligner:
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api
        self.shot_distance_descriptions = """
        Shot Distance Descriptions
        • Long Shot: Shows the relationship between characters and their environment, typically used to display the scene or environment.
        • Full Shot: Shows the full body of a character, commonly used to display movement or the full scene.
        • Medium Long Shot: Starts from above the character's knees, capturing part of the environment.
        • Medium Shot: Captures the character from the waist up.
        • Close-Up: Captures the character from the chest up.
        • Extreme Close-Up: Focuses on the character's head or face, with the background and environment typically blurred or not visible.
        """
        
        self.angle_descriptions = """
        Angle Descriptions
        • Eye Level Shot: The camera is positioned at the subject's eye level.
        • Low Angle Shot: The camera is positioned below eye level, shooting upward, emphasizing the character's power or size.
        • High Angle Shot: The camera is positioned above eye level, shooting downward, often minimizing the subject's significance.
        • Bird's Eye View: Camera shot taken from directly above, providing an overview of the scene.
        • Tilted Shot: The camera is intentionally tilted to create a sense of imbalance or tension.
        • Perspective Compression: A technique that emphasizes depth and the relationship between foreground and background through perspective.
        """
        
        # Alignment scoring tables
        self.distance_alignment_table = {
            "Long Shot": {"Long Shot": 5, "Full Shot": 4, "Medium Long Shot": 3, "Medium Shot": 2, "Close-Up": 1, "Extreme Close-Up": 1},
            "Full Shot": {"Long Shot": 4, "Full Shot": 5, "Medium Long Shot": 4, "Medium Shot": 3, "Close-Up": 1, "Extreme Close-Up": 1},
            "Medium Long Shot": {"Long Shot": 3, "Full Shot": 4, "Medium Long Shot": 5, "Medium Shot": 4, "Close-Up": 2, "Extreme Close-Up": 1},
            "Medium Shot": {"Long Shot": 2, "Full Shot": 3, "Medium Long Shot": 4, "Medium Shot": 5, "Close-Up": 4, "Extreme Close-Up": 2},
            "Close-Up": {"Long Shot": 1, "Full Shot": 1, "Medium Long Shot": 2, "Medium Shot": 4, "Close-Up": 5, "Extreme Close-Up": 4},
            "Extreme Close-Up": {"Long Shot": 1, "Full Shot": 1, "Medium Long Shot": 1, "Medium Shot": 2, "Close-Up": 4, "Extreme Close-Up": 5}
        }
        
        self.angle_alignment_table = {
            "Eye Level Shot": {"Eye Level Shot": 5, "Low Angle Shot": 2, "High Angle Shot": 2, "Bird's Eye View": 1, "Tilted Shot": 2, "Perspective Compression": 2},
            "Low Angle Shot": {"Eye Level Shot": 2, "Low Angle Shot": 5, "High Angle Shot": 2, "Bird's Eye View": 1, "Tilted Shot": 3, "Perspective Compression": 2},
            "High Angle Shot": {"Eye Level Shot": 2, "Low Angle Shot": 2, "High Angle Shot": 5, "Bird's Eye View": 4, "Tilted Shot": 3, "Perspective Compression": 2},
            "Bird's Eye View": {"Eye Level Shot": 1, "Low Angle Shot": 1, "High Angle Shot": 4, "Bird's Eye View": 5, "Tilted Shot": 2, "Perspective Compression": 2},
            "Tilted Shot": {"Eye Level Shot": 2, "Low Angle Shot": 3, "High Angle Shot": 3, "Bird's Eye View": 2, "Tilted Shot": 5, "Perspective Compression": 2},
            "Perspective Compression": {"Eye Level Shot": 2, "Low Angle Shot": 2, "High Angle Shot": 2, "Bird's Eye View": 2, "Tilted Shot": 2, "Perspective Compression": 5}
        }

        # Create results directory if it doesn't exist
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        self.results_dir = os.path.join(base_dir, "StoryVideoAlignment", "eval_res")
        os.makedirs(self.results_dir, exist_ok=True)
        self.results_file = os.path.join(self.results_dir, "ShotPerspectiveAlignment_results.json")

    def generate_shot_info(self, script_data: Dict) -> Dict:
        """Generate shot information for scripts without it"""
        prompt = f"""
        Based on the following script and character/environment descriptions, generate appropriate shot distance and angle information for each clip.

        {self.shot_distance_descriptions}

        {self.angle_descriptions}

        Script Data:
        {json.dumps(script_data, indent=2)}

        For each clip, analyze the content and suggest the most appropriate:
        1. Shot Distance (Long Shot, Full Shot, Medium Long Shot, Medium Shot, Close-Up, Extreme Close-Up)
        2. Camera Angle (Eye Level Shot, Low Angle Shot, High Angle Shot, Bird's Eye View, Tilted Shot, Perspective Compression)

        Return the result in JSON format with shot_distance and camera_angle fields added to each clip. Keep all original fields and just add the new ones.
        
        Example format:
        {{
            "characters": [...],
            "environments": [...],
            "script": [
                {{
                    "scene": 1,
                    "environment": 1,
                    "clips": [
                        {{
                            "id": 1,
                            "characters": ["1"],
                            "description": "...",
                            "shot_distance": "Close-Up",
                            "camera_angle": "Eye Level Shot"
                        }}
                    ]
                }}
            ]
        }}
        """
        
        print("Calling Gemini API to generate shot information...")
        
        # Retry logic for shot generation
        enhanced_script = script_data.copy()
        max_retries = 3
        retry_delay = 30  # 30 seconds
        
        for retry in range(max_retries):
            try:
                print(f"Shot generation attempt {retry + 1}/{max_retries}")
                response = self.gemini_api.generate_from_text(prompt)
                
                # Save the raw response for debugging
                self._save_intermediate_result(f"gemini_shot_generation_attempt_{retry + 1}", {
                    "attempt": retry + 1,
                    "prompt": prompt,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Try to parse the JSON response from Gemini
                try:
                    # Extract JSON from response if it's embedded in text
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        json_str = response[start_idx:end_idx]
                        gemini_result = json.loads(json_str)
                        
                        # Verify the structure and extract shot info
                        if "script" in gemini_result:
                            shots_added = 0
                            for scene_idx, scene in enumerate(enhanced_script["script"]):
                                if scene_idx < len(gemini_result["script"]):
                                    gemini_scene = gemini_result["script"][scene_idx]
                                    if "clips" in gemini_scene:
                                        for clip_idx, clip in enumerate(scene["clips"]):
                                            if clip_idx < len(gemini_scene["clips"]):
                                                gemini_clip = gemini_scene["clips"][clip_idx]
                                                # Extract shot info from Gemini's response
                                                if "shot_distance" in gemini_clip and "camera_angle" in gemini_clip:
                                                    clip["shot_distance"] = gemini_clip["shot_distance"]
                                                    clip["camera_angle"] = gemini_clip["camera_angle"]
                                                    shots_added += 1
                            
                            print(f"Successfully parsed shot information from Gemini response (added {shots_added} shots)")
                            self._save_intermediate_result("shot_generation_success", {
                                "parsed_successfully": True,
                                "shots_added": shots_added,
                                "attempt": retry + 1,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            return enhanced_script  # Success, return immediately
                        else:
                            print("Warning: Gemini response doesn't contain expected 'script' field")
                            raise ValueError("Invalid response structure")
                    else:
                        print("Warning: No JSON object found in Gemini response")
                        raise ValueError("No JSON found in response")
                        
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Error parsing Gemini response on attempt {retry + 1}: {e}")
                    if retry < max_retries - 1:
                        print(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                    continue
                    
            except Exception as e:
                print(f"API error on shot generation attempt {retry + 1}: {e}")
                if retry < max_retries - 1:
                    print(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                continue
        
        # If all retries failed, save failure record and return original script
        print(f"Failed to generate shot information after {max_retries} attempts")
        self._save_intermediate_result("shot_generation_failure", {
            "success": False,
            "reason": f"Failed after {max_retries} attempts",
            "timestamp": datetime.now().isoformat()
        })
        
        # Return original script without any shot information
        return script_data


    def _get_shot_files_in_order(self, shot_dir: str) -> List[str]:
        """Get shot files in alphabetical order"""
        import glob
        
        # Find all files in the directory
        pattern = os.path.join(shot_dir, "*")
        all_files = glob.glob(pattern)
        
        # Filter to only include files (not directories)
        shot_files = [f for f in all_files if os.path.isfile(f)]
        
        # Sort by filename in alphabetical order
        shot_files.sort()
        
        print(f"Found {len(shot_files)} shot files in alphabetical order:")
        for i, shot_file in enumerate(shot_files, 1):
            filename = os.path.basename(shot_file)
            print(f"  Shot {i}: {filename}")
        
        return shot_files

    def analyze_video_shots(self, shot_dir: str, script_data: Dict) -> List[Dict]:
        """Analyze actual shot distances and angles in individual shot files"""
        # Get ordered list of shot files
        shot_files = self._get_shot_files_in_order(shot_dir)
        
        if not shot_files:
            print(f"No shot files found in {shot_dir}")
            return []
        
        expected_shots = sum(len(scene['clips']) for scene in script_data['script'])
        print(f"Expected {expected_shots} shots, found {len(shot_files)} shot files")
        
        analyzed_shots = []
        
        for i, shot_file in enumerate(shot_files, 1):
            filename = os.path.basename(shot_file)
            print(f"Analyzing shot {i}/{len(shot_files)}: {filename}")
            
            prompt = f"""
            Analyze this single video shot and identify its shot distance and camera angle.

            {self.shot_distance_descriptions}

            {self.angle_descriptions}

            For this shot, determine:
            1. Shot Distance: Long Shot, Full Shot, Medium Long Shot, Medium Shot, Close-Up, or Extreme Close-Up
            2. Camera Angle: Eye Level Shot, Low Angle Shot, High Angle Shot, Bird's Eye View, Tilted Shot, or Perspective Compression

            Return your analysis in JSON format as a single object:
            {{
                "shot_number": {i},
                "shot_distance": "XXX Shot",
                "camera_angle": "XXX Shot",
                "description": "Brief description of what you see"
            }}
            """
            
            # Retry logic for this shot
            shot_analysis = None
            max_retries = 3
            retry_delay = 30  # 30 seconds
            
            for retry in range(max_retries):
                try:
                    print(f"  Attempt {retry + 1}/{max_retries}")
                    response = self.gemini_api.generate_from_videos([shot_file], prompt)
                    
                    # Only save the first attempt to reduce file size
                    if retry == 0:
                        self._save_intermediate_result(f"shot_{i}_analysis_attempt_1", {
                            "shot_file": shot_file,
                            "filename": filename,
                            "attempt": retry + 1,
                            "prompt": prompt,
                            "response": response,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    # Parse the JSON response
                    try:
                        # Extract JSON from response
                        start_idx = response.find('{')
                        end_idx = response.rfind('}') + 1
                        if start_idx != -1 and end_idx != 0:
                            json_str = response[start_idx:end_idx]
                            parsed_analysis = json.loads(json_str)
                            parsed_analysis["shot_number"] = i  # Ensure correct shot number
                            parsed_analysis["filename"] = filename
                            shot_analysis = parsed_analysis
                            print(f"  Successfully analyzed shot {i}")
                            break  # Success, exit retry loop
                        else:
                            raise ValueError("No JSON object found in response")
                    except json.JSONDecodeError as e:
                        print(f"  JSON parsing error on attempt {retry + 1}: {e}")
                        if retry < max_retries - 1:
                            print(f"  Waiting {retry_delay} seconds before retry...")
                            time.sleep(retry_delay)
                        continue
                    except ValueError as e:
                        print(f"  Response parsing error on attempt {retry + 1}: {e}")
                        if retry < max_retries - 1:
                            print(f"  Waiting {retry_delay} seconds before retry...")
                            time.sleep(retry_delay)
                        continue
                        
                except Exception as e:
                    print(f"  API error on attempt {retry + 1}: {e}")
                    if retry < max_retries - 1:
                        print(f"  Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                    continue
            
            # Add to results only if analysis was successful
            if shot_analysis:
                analyzed_shots.append(shot_analysis)
                # Save successful analysis
                self._save_intermediate_result(f"shot_{i}_final_analysis", {
                    "shot_analysis": shot_analysis,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                print(f"  Failed to analyze shot {i} ({filename}) after {max_retries} attempts. Skipping.")
                # Save failure record
                self._save_intermediate_result(f"shot_{i}_final_analysis", {
                    "shot_file": shot_file,
                    "filename": filename,
                    "success": False,
                    "reason": f"Failed after {max_retries} attempts",
                    "timestamp": datetime.now().isoformat()
                })
        
        print(f"Successfully analyzed {len(analyzed_shots)} out of {len(shot_files)} shots")
        
        # Save final parsed results
        self._save_intermediate_result("parsed_video_analysis", {
            "analyzed_shots": analyzed_shots,
            "total_shots_analyzed": len(analyzed_shots),
            "total_shots_attempted": len(shot_files),
            "success_rate": len(analyzed_shots) / len(shot_files) if shot_files else 0,
            "shot_files_processed": [os.path.basename(f) for f in shot_files],
            "timestamp": datetime.now().isoformat()
        })
        
        return analyzed_shots
    
    def calculate_alignment_score(self, expected_shots: List[Dict], actual_shots: List[Dict]) -> Dict:
        """Calculate alignment scores between expected and actual shots"""
        # Only calculate for shots that were successfully analyzed
        if len(actual_shots) == 0:
            print("Warning: No shots were successfully analyzed")
            return {
                "average_distance_score": 0,
                "average_angle_score": 0,
                "average_combined_score": 0,
                "total_shots": 0,
                "total_expected_shots": len(expected_shots),
                "success_rate": 0,
                "detailed_scores": []
            }
        
        if len(expected_shots) != len(actual_shots):
            print(f"Warning: Expected {len(expected_shots)} shots, but successfully analyzed {len(actual_shots)} shots")
            # Only process the number of shots we have
            min_length = min(len(expected_shots), len(actual_shots))
            expected_shots_subset = expected_shots[:min_length]
            actual_shots_subset = actual_shots[:min_length]
        else:
            expected_shots_subset = expected_shots
            actual_shots_subset = actual_shots
        
        distance_scores = []
        angle_scores = []
        detailed_scores = []
        
        for i, (expected, actual) in enumerate(zip(expected_shots_subset, actual_shots_subset)):
            # Get shot distance score
            expected_distance = expected.get("shot_distance", "Medium Shot")
            actual_distance = actual.get("shot_distance", "Medium Shot")
            distance_score = self.distance_alignment_table.get(expected_distance, {}).get(actual_distance, 1)
            
            # Get camera angle score
            expected_angle = expected.get("camera_angle", "Eye Level Shot")
            actual_angle = actual.get("camera_angle", "Eye Level Shot")
            angle_score = self.angle_alignment_table.get(expected_angle, {}).get(actual_angle, 1)
            
            distance_scores.append(distance_score)
            angle_scores.append(angle_score)
            
            detailed_scores.append({
                "shot_number": actual.get("shot_number", i + 1),
                "filename": actual.get("filename", f"shot_{i+1}"),
                "expected_distance": expected_distance,
                "actual_distance": actual_distance,
                "distance_score": distance_score,
                "expected_angle": expected_angle,
                "actual_angle": actual_angle,
                "angle_score": angle_score,
                "combined_score": (distance_score + angle_score) / 2
            })
        
        # Calculate averages
        avg_distance_score = sum(distance_scores) / len(distance_scores) if distance_scores else 0
        avg_angle_score = sum(angle_scores) / len(angle_scores) if angle_scores else 0
        avg_combined_score = (avg_distance_score + avg_angle_score) / 2
        
        return {
            "average_distance_score": avg_distance_score,
            "average_angle_score": avg_angle_score,
            "average_combined_score": avg_combined_score,
            "total_shots": len(actual_shots_subset),
            "total_expected_shots": len(expected_shots),
            "success_rate": len(actual_shots) / len(expected_shots) if expected_shots else 0,
            "detailed_scores": detailed_scores
        }

    def _save_intermediate_result(self, step_name: str, data: Dict):
        """Save intermediate results to JSON file - simplified version"""
        try:
            # Load existing results if file exists
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            else:
                results = {}
            
            # Only save essential information based on step type
            if "intermediate_steps" not in results:
                results["intermediate_steps"] = {}
            
            # Filter data based on step name to keep only essential info
            if step_name.startswith("gemini_shot_generation_attempt_"):
                filtered_data = {
                    "attempt": data.get("attempt", 0),
                    "prompt": data.get("prompt", ""),
                    "response": data.get("response", ""),
                    "timestamp": data.get("timestamp", "")
                }
            elif step_name == "shot_generation_success":
                filtered_data = {
                    "parsed_successfully": data.get("parsed_successfully", False),
                    "shots_added": data.get("shots_added", 0),
                    "attempt": data.get("attempt", 0),
                    "timestamp": data.get("timestamp", "")
                }
            elif step_name == "shot_generation_failure":
                filtered_data = {
                    "success": data.get("success", False),
                    "reason": data.get("reason", ""),
                    "timestamp": data.get("timestamp", "")
                }
            elif step_name.startswith("shot_") and step_name.endswith("_analysis_attempt_1"):
                # Only save the first attempt for each shot
                filtered_data = {
                    "filename": data.get("filename", ""),
                    "prompt": data.get("prompt", ""),
                    "response": data.get("response", ""),
                    "timestamp": data.get("timestamp", "")
                }
            elif step_name == "parsed_video_analysis":
                # Save shot analysis summary
                filtered_data = {
                    "total_shots_analyzed": data.get("total_shots_analyzed", 0),
                    "total_shots_attempted": data.get("total_shots_attempted", 0),
                    "success_rate": data.get("success_rate", 0),
                    "timestamp": data.get("timestamp", "")
                }
            elif step_name == "alignment_calculation":
                # Save alignment results summary
                alignment_results = data.get("alignment_results", {})
                filtered_data = {
                    "average_distance_score": alignment_results.get("average_distance_score", 0),
                    "average_angle_score": alignment_results.get("average_angle_score", 0),
                    "average_combined_score": alignment_results.get("average_combined_score", 0),
                    "total_shots": alignment_results.get("total_shots", 0),
                    "success_rate": alignment_results.get("success_rate", 0),
                    "timestamp": data.get("timestamp", "")
                }
            else:
                # For other steps, skip saving
                return
            
            results["intermediate_steps"][step_name] = filtered_data
            
            # Save back to file
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving intermediate result: {e}")

    def _save_final_results(self, results: Dict):
        """Save final results to JSON file - simplified version"""
        try:
            # Load existing results if file exists
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            else:
                existing_results = {}
            
            # Create simplified shot information
            shot_comparisons = []
            alignment_results = results.get('alignment_results', {})
            detailed_scores = alignment_results.get('detailed_scores', [])
            
            for shot_detail in detailed_scores:
                shot_comparisons.append({
                    "shot_number": shot_detail.get("shot_number", 0),
                    "filename": shot_detail.get("filename", ""),
                    "expected": {
                        "shot_distance": shot_detail.get("expected_distance", ""),
                        "camera_angle": shot_detail.get("expected_angle", "")
                    },
                    "actual": {
                        "shot_distance": shot_detail.get("actual_distance", ""),
                        "camera_angle": shot_detail.get("actual_angle", "")
                    },
                    "scores": {
                        "distance_score": shot_detail.get("distance_score", 0),
                        "angle_score": shot_detail.get("angle_score", 0),
                        "combined_score": shot_detail.get("combined_score", 0)
                    }
                })
            
            # Add simplified final results
            simplified_results = {
                "shot_comparisons": shot_comparisons,
                "summary": {
                    "total_shots_analyzed": alignment_results.get("total_shots", 0),
                    "total_expected_shots": alignment_results.get("total_expected_shots", 0),
                    "success_rate": alignment_results.get("success_rate", 0),
                    "average_distance_score": alignment_results.get("average_distance_score", 0),
                    "average_angle_score": alignment_results.get("average_angle_score", 0),
                    "overall_alignment_score": alignment_results.get("average_combined_score", 0)
                },
                "completion_timestamp": datetime.now().isoformat()
            }
            
            existing_results["final_results"] = simplified_results
            
            # Save to file
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to: {self.results_file}")
                
        except Exception as e:
            print(f"Error saving final results: {e}")

    def process_script_and_video(self, script_data: Dict, shot_dir: str) -> Dict:
        """Main method to process script and shot directory for shot alignment analysis"""
        print("Step 1: Processing script data...")
        
        # Save initial input data (excluding this from simplified saving)
        
        # Check if script already has shot information
        has_shot_info = False
        for scene in script_data["script"]:
            for clip in scene["clips"]:
                if "shot_distance" in clip and "camera_angle" in clip:
                    has_shot_info = True
                    break
            if has_shot_info:
                break
        
        print(f"Script has shot info: {has_shot_info}")
        
        # Generate shot info if not present
        if not has_shot_info:
            print("No shot information found. Generating shot information...")
            enhanced_script = self.generate_shot_info(script_data)
            
            # Check if shot generation was successful
            shots_with_info = 0
            total_shots = 0
            for scene in enhanced_script["script"]:
                for clip in scene["clips"]:
                    total_shots += 1
                    if "shot_distance" in clip and "camera_angle" in clip:
                        shots_with_info += 1
            
            if shots_with_info == 0:
                print("ERROR: No shot information could be generated. Cannot proceed with analysis.")
                return {
                    "error": "Shot generation failed",
                    "enhanced_script": enhanced_script,
                    "expected_shots": [],
                    "actual_shots": [],
                    "alignment_results": {
                        "average_distance_score": 0,
                        "average_angle_score": 0,
                        "average_combined_score": 0,
                        "total_shots": 0,
                        "total_expected_shots": total_shots,
                        "success_rate": 0,
                        "detailed_scores": []
                    }
                }
            
            print(f"Generated shot info for {shots_with_info}/{total_shots} shots")
        else:
            print("Shot information found in script.")
            enhanced_script = script_data
        
        # Extract expected shots (only include clips with complete shot info)
        expected_shots = []
        for scene in enhanced_script["script"]:
            for clip in scene["clips"]:
                if "shot_distance" in clip and "camera_angle" in clip:
                    expected_shots.append({
                        "shot_distance": clip["shot_distance"],
                        "camera_angle": clip["camera_angle"],
                        "description": clip.get("description", "")
                    })
        
        if len(expected_shots) == 0:
            print("ERROR: No shots with complete information available for analysis.")
            return {
                "error": "No complete shot information",
                "enhanced_script": enhanced_script,
                "expected_shots": [],
                "actual_shots": [],
                "alignment_results": {
                    "average_distance_score": 0,
                    "average_angle_score": 0,
                    "average_combined_score": 0,
                    "total_shots": 0,
                    "total_expected_shots": 0,
                    "success_rate": 0,
                    "detailed_scores": []
                }
            }
        
        print(f"Step 2: Analyzing shot directory with {len(expected_shots)} expected shots...")
        actual_shots = self.analyze_video_shots(shot_dir, enhanced_script)
        
        print("Step 3: Calculating alignment scores...")
        alignment_results = self.calculate_alignment_score(expected_shots, actual_shots)
        
        # Save alignment calculation
        self._save_intermediate_result("alignment_calculation", {
            "alignment_results": alignment_results,
            "timestamp": datetime.now().isoformat()
        })
        
        final_results = {
            "enhanced_script": enhanced_script,
            "expected_shots": expected_shots,
            "actual_shots": actual_shots,
            "alignment_results": alignment_results
        }
        
        # Save final results
        self._save_final_results(final_results)
        
        return final_results


# Example usage
if __name__ == "__main__":
    # Initialize Gemini API
    gemini = GeminiAPI(
        api_keys=["YOUR_GEMINI_API_KEY", 
                 "YOUR_GEMINI_API_KEY", 
                 "YOUR_GEMINI_API_KEY"],
        proxy="YOUR_PROXY_URL"  # Set to None if no proxy is needed
    )
    
    # Initialize shot perspective aligner
    aligner = ShotPerspectiveAligner(gemini)
    
    # Shot directory
    shot_dir = "path/to/your/shot_dir"  # Replace with an actual shot directory.
    
    # Script data
    script_data = {
        "characters": "your characters here",
        "environments": "your environments here",
        "script": "your script data here"
    }

    # Process script and shot directory
    results = aligner.process_script_and_video(script_data, shot_dir)
    
    # Print results
    print("\n=== SHOT PERSPECTIVE ALIGNMENT RESULTS ===")
    print(f"Total shots analyzed: {results['alignment_results']['total_shots']}")
    print(f"Average distance score: {results['alignment_results']['average_distance_score']:.2f}/5.0")
    print(f"Average angle score: {results['alignment_results']['average_angle_score']:.2f}/5.0")
    print(f"Overall alignment score: {results['alignment_results']['average_combined_score']:.2f}/5.0")
    
    print("\n=== DETAILED SHOT ANALYSIS ===")
    for shot in results['alignment_results']['detailed_scores']:
        print(f"Shot {shot['shot_number']}:")
        print(f"  Distance: {shot['expected_distance']} → {shot['actual_distance']} (Score: {shot['distance_score']}/5)")
        print(f"  Angle: {shot['expected_angle']} → {shot['actual_angle']} (Score: {shot['angle_score']}/5)")
        print(f"  Combined Score: {shot['combined_score']:.2f}/5")
        print()
    
    print(f"\nAll results saved to: {aligner.results_file}")