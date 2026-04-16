"""
Video detection count scoring system.
Uses Gemini to detect and score video content.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
TOOLS_DIR = os.path.join(BASE_DIR, "Tools")
OUTPUT_DIR = os.path.join(BASE_DIR, "StoryVideoAlignment", "gemini_detection_results")

sys.path.append(TOOLS_DIR)
from gemini_api import GeminiAPI

class VideoDetectionEvaluator:
    """Video detection evaluator."""
    
    # Configuration constants
    GEMINI_API_KEYS = [
        # "YOUR_GEMINI_API_KEY", 
        # "YOUR_GEMINI_API_KEY", 
        # "YOUR_GEMINI_API_KEY"
        "YOUR_GEMINI_API_KEY"
    ]
    PROXY = "YOUR_PROXY_URL"  # Set to None if no proxy is needed
    
    # Gemini prompt templates
    DETECTION_REQUIREMENTS_PROMPT = """
    Based on the following story description, please analyze the key objects and characters that can been seen in the video, and categorize them into two types:

    Story description: {description}

    Important instructions:
    1. When referring to people/characters, DO NOT use specific names like "Lily" or general terms like "mom", "mommy", "dad", etc.
    2. Instead, describe people by their clothes color based on the following character information, such as "a girl in black", "a woman in blue", "a man in black", etc. If there are no corresponding character information, use general terms like "a girl", "a woman", etc.
    Character information: {character}

    Please output in JSON format, including the following two categories:
    1. "full_presence": Objects/characters that need to appear throughout the entire video (such as main characters, background items)
    2. "any_presence": Objects/characters that only need to appear at any point in the video (such as action-related objects)

    Output format example:
    {{
        "full_presence": ["a girl in black", "a golden dog", "green book"],
        "any_presence": ["table", "red ball"]
    }}

    Please ensure:
    - Object/character names are concise and clear
    - Use visual appearance descriptions for people instead of names or family relationships
    - Categorize by importance
    - Only output JSON, no other explanations
    """

    VIDEO_ANALYSIS_PROMPT = """
    Please analyze this video and identify which of the following objects/characters are present:

    Required objects/characters to detect:
    {objects_list}

    For each object/character in the list, please determine:
    1. Whether it appears in the video (yes/no)
    2. If it appears, rate its presence on a scale of 0-5:
       - 0: Not present at all throughout the video
       - 1: Appears very briefly (less than 20% of video duration)
       - 2: Appears occasionally (20-40% of video duration)
       - 3: Appears moderately (40-60% of video duration)
       - 4: Appears frequently (60-80% of video duration)
       - 5: Appears throughout most/all of the video (80-100% of video duration)

    Please provide your analysis in the following JSON format:
    {{
        "detections": {{
            "object_name_1": {{"present": true/false, "presence_score": 0-5}},
            "object_name_2": {{"present": true/false, "presence_score": 0-5}},
            ...
        }}
    }}

    Be thorough in your analysis and only mark objects as present if you can clearly see them in the video.
    Only output JSON, no other explanations.
    """
    
    def __init__(self, character: str, description: str, video_path: str):
        """
        Initialize the evaluator
        
        Args:
            character: Character information
            description: Story description
            video_path: Video file path
        """
        self.character = character
        self.description = description
        self.video_path = video_path
        
        # Initialize Gemini API
        self.gemini = GeminiAPI(api_keys=self.GEMINI_API_KEYS, proxy=self.PROXY)
        
        # Set output directory
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_detection_requirements(self) -> Dict[str, List[str]]:
        """
        Use Gemini to generate detection requirements from character and story description
        
        Returns:
            dict: Dictionary containing the keys 'full_presence' and 'any_presence'
        """
        prompt = self.DETECTION_REQUIREMENTS_PROMPT.format(
            character=self.character, 
            description=self.description
        )
        
        print("Generating detection requirements with Gemini...")
        response = self.gemini.generate_from_text(prompt)
        print(f"Gemini response: {response}")
        
        try:
            # Clean response formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            requirements = json.loads(cleaned_response)
            if "full_presence" not in requirements or "any_presence" not in requirements:
                raise ValueError("Invalid response format")
            
            # Save results
            result_data = {
                "raw_response": response,
                "cleaned_response": cleaned_response,
                "parsed_requirements": requirements,
                "character": self.character,
                "description": self.description,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            video_name = Path(self.video_path).stem
            output_path = self.output_dir / f"requirements_{video_name}_{int(time.time())}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"Detection requirements saved to: {output_path}")
            return requirements
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse Gemini response: {e}")
            return {"full_presence": [], "any_presence": []}
    
    def analyze_video_with_gemini(self, requirements: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Use Gemini to analyze object detections in the video
        
        Args:
            requirements: Detection requirements
            
        Returns:
            Dict: Detection results
        """
        # Merge all objects that need to be detected
        all_objects = requirements["full_presence"] + requirements["any_presence"]
        objects_list = "\n".join([f"- {obj}" for obj in all_objects])
        
        prompt = self.VIDEO_ANALYSIS_PROMPT.format(objects_list=objects_list)
        
        print("Analyzing video with Gemini...")
        print(f"Objects to detect: {all_objects}")
        
        # Upload and analyze video
        video_file = self.gemini.upload_video(self.video_path)
        response = self.gemini.generate_from_video_file(video_file, prompt)
        
        print(f"Gemini video analysis response: {response}")
        
        try:
            # Clean response formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            analysis_result = json.loads(cleaned_response)
            
            # Save analysis results
            result_data = {
                "raw_response": response,
                "cleaned_response": cleaned_response,
                "analysis_result": analysis_result,
                "requirements": requirements,
                "video_path": self.video_path,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            video_name = Path(self.video_path).stem
            output_path = self.output_dir / f"analysis_{video_name}_{int(time.time())}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"Video analysis results saved to: {output_path}")
            return analysis_result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse Gemini video analysis response: {e}")
            return {"detections": {}}
    
    def calculate_final_score(self, analysis_result: Dict[str, Any], requirements: Dict[str, List[str]]) -> float:
        """
        Calculate final score from analysis results
        
        Args:
            analysis_result: Gemini analysis result
            requirements: Detection requirements
            
        Returns:
            float: Final score
        """
        detections = analysis_result.get("detections", {})
        scores = {}
        
        # Process objects that must appear throughout the entire video
        for obj in requirements["full_presence"]:
            detection = detections.get(obj, {"present": False, "presence_score": 0})
            if detection["present"]:
                # Compute score on a 0-5 scale and normalize to 0-1
                score = detection["presence_score"] / 5.0
            else:
                score = 0.0
            scores[obj] = score
            print(f"Full-presence object '{obj}': present={detection['present']}, presence_score={detection.get('presence_score', 0)}/5, score={score:.3f}")
        
        # Process objects that only need to appear at any point
        for obj in requirements["any_presence"]:
            detection = detections.get(obj, {"present": False, "presence_score": 0})
            # Full score if present, otherwise 0
            score = 1.0 if detection["present"] else 0.0
            scores[obj] = score
            print(f"Any-presence object '{obj}': present={detection['present']}, score={score:.3f}")
        
        # Compute average score
        if scores:
            average_score = sum(scores.values()) / len(scores)
        else:
            average_score = 0.0
        
        # Save scoring results
        score_data = {
            "scores": scores,
            "average_score": average_score,
            "requirements": requirements,
            "analysis_result": analysis_result,
            "video_path": self.video_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        video_name = Path(self.video_path).stem
        output_path = self.output_dir / f"final_score_{video_name}_{int(time.time())}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(score_data, f, indent=2, ensure_ascii=False)
        
        print(f"Final scoring results saved to: {output_path}")
        return average_score
    
    def evaluate(self) -> float:
        """
        Evaluate video detection score
        
        Returns:
            float: Average score
        """
        print("=" * 50)
        print("Starting video detection scoring (Gemini-only version)")
        print(f"Character info: {self.character}")
        print(f"Description: {self.description}")
        print(f"Video: {self.video_path}")
        print("=" * 50)
        
        try:
            # 1. Generate detection requirements
            requirements = self.generate_detection_requirements()
            if not requirements["full_presence"] and not requirements["any_presence"]:
                print("Failed to generate valid detection requirements")
                return 0.0
            
            # 2. Analyze the video with Gemini
            analysis_result = self.analyze_video_with_gemini(requirements)
            if not analysis_result.get("detections"):
                print("Failed to get valid video analysis results")
                return 0.0
            
            # 3. Calculate final score
            final_score = self.calculate_final_score(analysis_result, requirements)
            
            print(f"\nFinal detection score: {final_score:.3f}")
            print("=" * 50)
            return final_score
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 0.0

if __name__ == "__main__":
    # Example data
    character = "your character information here"
    description = "your story description here"  # Replace with an actual story description.
    video_path = "path/to/your/video.mp4"  # Replace with an actual video path.
    
    # Create evaluator and run evaluation
    evaluator = VideoDetectionEvaluator(character, description, video_path)
    score = evaluator.evaluate()
    print(f"\nFinal detection score: {score:.3f}")