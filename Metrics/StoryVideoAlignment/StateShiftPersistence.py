import json
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
TOOLS_DIR = os.path.join(BASE_DIR, "Tools")
EVAL_RES_DIR = os.path.join(BASE_DIR, "StoryVideoAlignment", "eval_res")

sys.path.append(TOOLS_DIR)
from gemini_api import GeminiAPI

class VideoChangeAnalyzer:
    def __init__(self, api_keys, proxy=None):
        self.gemini = GeminiAPI(api_keys=api_keys, proxy=proxy)
        self.current_video_file = None  # Store uploaded video file
        
    def analyze_script_changes(self, script_data):
        """Step 1: Analyze script and identify state-change points."""
        script_json = json.dumps(script_data, indent=2, ensure_ascii=False)
        
        prompt = f"""
        Analyze the following script to identify change points where character states, environment states, or other important elements undergo internal transitions.

        IMPORTANT: Focus on INTERNAL changes within existing characters/environments, NOT switching between different characters or scenes. Examples:
        - Character changes: clothing getting dirty/torn, changing clothes, getting injured, hair getting wet, etc.
        - Environment changes: sunny to rainy weather, day to night, windows breaking, objects being consumed/destroyed, furniture moving, lighting changes, etc.
        - DO NOT include: characters motion changes, new characters entering, scene transitions, camera angle changes, or switching between different locations.

        Script content:
        {script_json}

        Please output change points in the following format:
        1. Change type (character change/environment change/other)
        2. Specific description of the internal state change
        3. State before the change
        4. State after the change

        Please output in JSON format, for example:
        {{
            "changes": [
                {{
                    "type": "character change",
                    "description": "Lily's clothes get crumbs on them",
                    "before_state": "Lily has clean clothes",
                    "after_state": "Lily has crumbs on her clothes"
                }}
            ]
        }}
        """
        
        response = self.gemini.generate_from_text(prompt)
        print("=== Step 1: Script Analysis Result ===")
        print(response)
        
        # Convert JSON string to proper JSON object
        try:
            if isinstance(response, str):
                # Try to extract JSON from response if it's wrapped in markdown
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    return json.loads(json_str)
                else:
                    return json.loads(response)
            return response
        except json.JSONDecodeError:
            print("Warning: Could not parse JSON response, returning raw response")
            return response
    
    def locate_changes_in_video(self, movie_path, script_changes):
        """Step 2: Locate change points in video and assign initial 1-2 or 3-5 scores."""
        # Upload video if not already uploaded or if it's a different video
        if self.current_video_file is None or movie_path not in self.gemini.uploaded_videos:
            self.current_video_file = self.gemini.upload_video(movie_path)
        else:
            self.current_video_file = self.gemini.uploaded_videos[movie_path]
        
        # Convert script_changes to JSON string if it's an object
        if isinstance(script_changes, dict):
            script_changes_str = json.dumps(script_changes, ensure_ascii=False, indent=2)
        else:
            script_changes_str = script_changes
            
        prompt = f"""
        Based on the following script analysis results, locate these changes in the video and provide initial scoring.

        Script change analysis:
        {script_changes_str}

        For each change point, please:
        1. Determine if the change can be observed in the video
        2. If observable, provide the approximate timestamp (seconds) when the change occurs
        3. Describe the specific manifestation of the change in the video
        4. Provide initial scoring based on change detection:
        - Score 1: Change is completely not reflected in the video
        - Score 2: Change moment is vague and cannot be accurately located
        - Score 3-5: Change is clearly detectable (will be further evaluated in next step)

        Please output in JSON format, for example:
        {{
            "located_changes": [
                {{
                    "change_id": 1,
                    "found": true,
                    "timestamp": "12.5 seconds",
                    "video_description": "At 12.5 seconds, Lily can be seen picking up the bread",
                    "initial_score": "3-5",
                    "score_reason": "Change is clearly visible and can be accurately located"
                }},
                {{
                    "change_id": 2,
                    "found": false,
                    "initial_score": 1,
                    "score_reason": "Change is completely not reflected in the video"
                }}
            ]
        }}
        """
        
        response = self.gemini.generate_from_video_file(self.current_video_file, prompt)
        print("=== Step 2: Video Change Location Result ===")
        print(response)
        
        # Convert JSON string to proper JSON object
        try:
            if isinstance(response, str):
                # Try to extract JSON from response if it's wrapped in markdown
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    return json.loads(json_str)
                else:
                    return json.loads(response)
            return response
        except json.JSONDecodeError:
            print("Warning: Could not parse JSON response, returning raw response")
            return response
    
    def evaluate_change_persistence(self, movie_path, located_changes):
        """Step 3: Evaluate persistence for 3-5 score changes while preserving 1-2 score changes."""
        # Use the already uploaded video file
        if self.current_video_file is None:
            self.current_video_file = self.gemini.upload_video(movie_path)
        
        # Convert located_changes to JSON string if it's an object
        if isinstance(located_changes, dict):
            located_changes_str = json.dumps(located_changes, ensure_ascii=False, indent=2)
        else:
            located_changes_str = located_changes
            
        prompt = f"""
        Based on the following change location results, evaluate the persistence and consistency of changes that scored 3-5 in the video, and preserve the scores of changes that scored 1-2.

        Change location results:
        {located_changes_str}

        For each change that received a score of 3-5 (clearly detectable), please analyze:
        1. After the change occurs, is the new state maintained consistently in subsequent segments?
        2. Are there any shots that violate the post-change state?
        3. What is the degree of change persistence?

        For changes that received scores of 1-2, simply preserve their original scores and reasons.

        Scoring criteria for detectable changes:
        - Score 5: Subsequent segments consistently maintain the change until video end or new change occurs
        - Score 4: Subsequent segments maintain part of the change, or some shots preserve the change
        - Score 3: Subsequent segments completely do not maintain the change
        - Score 1-2: Keep original scores from location step

        Please output in JSON format, for example:
        {{
            "all_evaluations": [
                {{
                    "change_id": 1,
                    "final_score": 5,
                    "evaluation_type": "persistence_evaluated",
                    "explanation": "After Lily gets the bread, she can be seen holding or eating bread in all subsequent shots",
                    "evidence": "From 12.5 seconds until scene end, Lily is consistently handling the bread"
                }},
                {{
                    "change_id": 2,
                    "final_score": 1,
                    "evaluation_type": "location_failed",
                    "explanation": "Change is completely not reflected in the video",
                    "evidence": "Original score from location step"
                }}
            ]
        }}
        """
        
        response = self.gemini.generate_from_video_file(self.current_video_file, prompt)
        print("=== Step 3: Change Persistence Evaluation ===")
        print(response)
        
        # Convert JSON string to proper JSON object
        try:
            if isinstance(response, str):
                # Try to extract JSON from response if it's wrapped in markdown
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                    return json.loads(json_str)
                else:
                    return json.loads(response)
            return response
        except json.JSONDecodeError:
            print("Warning: Could not parse JSON response, returning raw response")
            return response
    
    def calculate_final_score(self, located_changes, persistence_evaluation):
        """Step 4: Compute the final composite score directly, without extra LMM interaction."""
        try:
            # Parse persistence evaluation results - handle both dict and string inputs
            if isinstance(persistence_evaluation, dict):
                persistence_data = persistence_evaluation
            else:
                persistence_data = json.loads(persistence_evaluation)
            
            # Collect all scores
            all_scores = []
            score_details = []
            
            # Process unified evaluation results
            for evaluation in persistence_data.get("all_evaluations", []):
                final_score = evaluation.get("final_score")
                if final_score:
                    all_scores.append(final_score)
                    score_details.append({
                        "change_id": evaluation.get("change_id"),
                        "score": final_score,
                        "type": evaluation.get("evaluation_type", "unknown"),
                        "explanation": evaluation.get("explanation", ""),
                        "evidence": evaluation.get("evidence", "")
                    })
            
            # Fall back to legacy logic if all_evaluations is empty
            if not all_scores:
                # Parse location results
                if isinstance(located_changes, dict):
                    located_data = located_changes
                else:
                    located_data = json.loads(located_changes)
                
                # Process 1-2 score changes (not successfully located)
                for change in located_data.get("located_changes", []):
                    if change.get("initial_score") in [1, 2]:
                        all_scores.append(change["initial_score"])
                        score_details.append({
                            "change_id": change.get("change_id"),
                            "score": change["initial_score"],
                            "type": "location_failed",
                            "reason": change.get("score_reason", "")
                        })
                
                # Process 3-5 score changes (located and persistence-evaluated)
                for evaluation in persistence_data.get("persistence_evaluation", []):
                    final_score = evaluation.get("final_score")
                    if final_score:
                        all_scores.append(final_score)
                        score_details.append({
                            "change_id": evaluation.get("change_id"),
                            "score": final_score,
                            "type": "persistence_evaluated",
                            "explanation": evaluation.get("explanation", "")
                        })
            
            # Compute average score
            if all_scores:
                final_score = sum(all_scores) / len(all_scores)
            else:
                final_score = 0
            
            # Summarize score distribution
            score_breakdown = {
                "score_1": sum(1 for s in all_scores if s == 1),
                "score_2": sum(1 for s in all_scores if s == 2),
                "score_3": sum(1 for s in all_scores if s == 3),
                "score_4": sum(1 for s in all_scores if s == 4),
                "score_5": sum(1 for s in all_scores if s == 5)
            }
            
            result = {
                "final_score": round(final_score, 2),
                "total_changes": len(all_scores),
                "individual_scores": all_scores,
                "score_breakdown": score_breakdown,
                "score_details": score_details,
                "overall_assessment": f"Video achieved an average score of {final_score:.2f} for state change detection and persistence across {len(all_scores)} identified changes."
            }
            
            print("=== Step 4: Final Comprehensive Score ===")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {"error": "Failed to parse evaluation results", "final_score": 0}
        except Exception as e:
            print(f"Error calculating final score: {e}")
            return {"error": str(e), "final_score": 0}
    
    def full_analysis(self, movie_path, script_data):
        """Complete analysis workflow."""
        print("Starting video state-change and persistence analysis...")
        
        # Step 1: Analyze state changes in the script
        res1 = self.analyze_script_changes(script_data)
        
        # Step 2: Locate changes in video and assign initial scores
        res2 = self.locate_changes_in_video(movie_path, res1)
        
        # Step 3: Evaluate persistence only for locatable changes
        res3 = self.evaluate_change_persistence(movie_path, res2)
        
        # Step 4: Compute final score directly
        final_res = self.calculate_final_score(res2, res3)
        
        # Print final total score
        final_score = final_res.get("final_score", 0)
        print(f"\n=== Final total score: {final_score} ===")
        
        return {
            "script_analysis": res1,
            "video_location": res2,
            "persistence_evaluation": res3,
            "final_result": final_res
        }

# Example usage and test data
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = VideoChangeAnalyzer(
        api_keys=["YOUR_GEMINI_API_KEY", 
                 "YOUR_GEMINI_API_KEY", 
                 "YOUR_GEMINI_API_KEY"],
        proxy="YOUR_PROXY_URL"  # Set to None if no proxy is needed
    )
    
    # Video path
    movie_path = "path/to/your/video.mp4"  # Replace with an actual video path.
    
    # Script data
    script_data = {
        "characters": "your characters here",
        "environments": "your environments here",
        "script": "your script data here"
    }

    # Run full analysis
    results = analyzer.full_analysis(movie_path, script_data)
    
    # Save results to file
    output_path = os.path.join(EVAL_RES_DIR, "StateShiftPersistence_results.json")
    os.makedirs(EVAL_RES_DIR, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to: {output_path}")