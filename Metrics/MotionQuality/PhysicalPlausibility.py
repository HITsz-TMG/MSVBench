import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.gemini_api import GeminiAPI
from typing import Dict, Any, List, Optional


class PhysicalPlausibility:
    def __init__(self, gemini_api: GeminiAPI):
        """
        Initialize PhysicalPlausibility evaluator
        
        Args:
            gemini_api: Instance of GeminiAPI for video analysis
        """
        self.gemini_api = gemini_api
        self.scoring_criteria = {
            0: "No characters found in video",
            1: "Characters present but not moving",
            2: "Characters moving but mostly violates physics", 
            3: "Characters moving with partially correct physics",
            4: "Characters moving with mostly correct physics",
            5: "Characters moving with completely realistic physics"
        }
    
    def analyze_expected_characters(self, prompt: str) -> Dict[str, Any]:
        """
        Step 1: Analyze prompt to identify all moving characters and their expected motions
        
        Args:
            prompt: Description of the expected scene
            
        Returns:
            Dictionary containing analysis results for all characters
        """
        analysis_prompt = f"""
        Analyze the following prompt and identify ALL moving characters and their expected motions:
        
        Prompt: "{prompt}"
        
        Please identify:
        1. List ALL characters that should be moving (humans, animals, creatures, etc.)
        2. For each character, describe their expected motion type and physical requirements
        3. Key physical principles that should govern each character's movement
        
        Format your response as:
        CHARACTER_COUNT: [total number of moving characters]
        CHARACTER_1: [name/type of first character]
        MOTION_1: [describe expected motion and physical requirements]
        PHYSICS_1: [key physical laws that should apply]
        CHARACTER_2: [name/type of second character]
        MOTION_2: [describe expected motion and physical requirements]
        PHYSICS_2: [key physical laws that should apply]
        [Continue for all characters...]
        
        Focus on realistic movement patterns, balance, gravity effects, momentum, etc.
        If no moving characters are expected, state CHARACTER_COUNT: 0
        """
        
        response = self.gemini_api.generate_from_text(analysis_prompt)
        return self._parse_character_analysis(response)

    def evaluate_character_physics(self, video_path: str, expected_characters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Evaluate physics accuracy for each character in the video
        
        Args:
            video_path: Path to the video file
            expected_characters: Results from step 1 analysis
            
        Returns:
            Dictionary containing evaluation results for each character
        """
        if not expected_characters or expected_characters.get('character_count', 0) == 0:
            return {"overall_score": 0, "reason": "No moving characters expected"}
        
        character_count = expected_characters.get('character_count', 0)
        character_evaluations = []
        
        for i in range(1, character_count + 1):
            character_key = f'character_{i}'
            motion_key = f'motion_{i}'
            physics_key = f'physics_{i}'
            
            if character_key in expected_characters:
                evaluation = self._evaluate_single_character(
                    video_path,
                    expected_characters[character_key],
                    expected_characters.get(motion_key, ''),
                    expected_characters.get(physics_key, '')
                )
                character_evaluations.append(evaluation)
        
        # Calculate average score
        if character_evaluations:
            scores = [eval_result.get('score', 0) for eval_result in character_evaluations]
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = 0
        
        return {
            "overall_score": round(overall_score, 2),
            "character_evaluations": character_evaluations,
            "character_count": len(character_evaluations)
        }

    def _evaluate_single_character(self, video_path: str, character: str, expected_motion: str, physics_principles: str) -> Dict[str, Any]:
        """
        Evaluate physics accuracy for a single character
        
        Args:
            video_path: Path to the video file
            character: Character description
            expected_motion: Expected motion description
            physics_principles: Relevant physics principles
            
        Returns:
            Dictionary containing evaluation results for this character
        """
        evaluation_prompt = f"""
        Analyze this video focusing on the physical plausibility of this specific character's movement:
        
        Character: {character}
        Expected Motion: {expected_motion}
        Physics Principles: {physics_principles}
        
        Evaluate how well this character's movement follows realistic physics:
        1. Is the character "{character}" present and moving in the video?
        2. Does their movement respect gravity, momentum, and balance?
        3. Are their body mechanics and motion dynamics realistic?
        4. Do they move in a way that follows natural physics laws?
        
        Give a score from 0-5 based on these detailed criteria:
        
        0 = Character not found in video
        - The specified character is completely absent from the video
        - Cannot locate the character in any frame
        
        1 = Character present but not moving
        - The specified character is visible but completely static
        - No movement to evaluate for physics
        
        2 = Character moving but mostly violates physics
        - Movement violates basic physical laws (gravity, momentum, etc.)
        - Character floats, moves through solid objects, or defies physics
        - Motion is mostly unrealistic and impossible
        
        3 = Character moving with partially correct physics
        - Basic physics followed but with noticeable inaccuracies
        - Some movements realistic, others clearly wrong
        - Mixed realistic and unrealistic motion patterns
        
        4 = Character moving with mostly correct physics
        - Movement is largely realistic with minor physics issues
        - Generally natural motion with small deviations
        - Overall convincing but some details feel slightly off
        
        5 = Character moving with completely realistic physics
        - All movement follows natural physics perfectly
        - Realistic body mechanics, gravity effects, momentum
        - Motion appears completely natural and believable
        
        Format response as:
        CHARACTER_PRESENT: [Yes/No]
        MOVEMENT_ANALYSIS: [detailed analysis of the character's movement physics]
        SCORE: [0-5]
        JUSTIFICATION: [explanation for the score with specific physics observations]
        """
        
        try:
            response = self.gemini_api.generate_from_videos([video_path], evaluation_prompt)
            results = self._parse_character_evaluation(response)
            
            return {
                "character": character,
                "score": results.get('score', 0),
                "present": results.get('character_present', False),
                "movement_analysis": results.get('movement_analysis', ''),
                "justification": results.get('justification', '')
            }
            
        except Exception as e:
            print(f"Error evaluating character {character}: {e}")
            return {
                "character": character,
                "score": 0,
                "present": False,
                "movement_analysis": "",
                "justification": f"Evaluation error: {str(e)}"
            }

    def score_physical_plausibility(self, video_path: str, prompt: str) -> Dict[str, Any]:
        """
        Complete evaluation pipeline for physical plausibility
        
        Args:
            video_path: Path to the video file
            prompt: Description of expected scene
            
        Returns:
            Dictionary containing complete evaluation results
        """
        print("Step 1: Analyzing expected characters and motions from prompt...")
        expected_characters = self.analyze_expected_characters(prompt)
        print(f"Expected characters: {expected_characters}")
        
        print("Step 2: Evaluating physics accuracy for each character...")
        evaluation_results = self.evaluate_character_physics(video_path, expected_characters)
        print(f"Evaluation results: {evaluation_results}")
        
        final_score = evaluation_results.get('overall_score', 0)
        
        return {
            "video_path": video_path,
            "prompt": prompt,
            "expected_characters": expected_characters,
            "evaluation_results": evaluation_results,
            "final_score": final_score,
            "score_description": self.scoring_criteria.get(int(final_score), "Unknown score"),
            "detailed_analysis": {
                "step1_analysis": expected_characters,
                "step2_evaluation": evaluation_results
            }
        }
    
    def _parse_character_analysis(self, response: str) -> Dict[str, Any]:
        """Parse the character analysis response"""
        result = {}
        lines = response.split('\n')
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('CHARACTER_COUNT:'):
                count_text = line.replace('CHARACTER_COUNT:', '').strip()
                try:
                    result['character_count'] = int(count_text)
                except:
                    result['character_count'] = 0
            elif line.startswith('CHARACTER_'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    result[key] = value
                    current_section = key
            elif line.startswith('MOTION_'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    result[key] = value
                    current_section = key
            elif line.startswith('PHYSICS_'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    result[key] = value
                    current_section = key
            elif current_section and line and not line.startswith(('CHARACTER_', 'MOTION_', 'PHYSICS_')):
                result[current_section] += ' ' + line
        
        return result
    
    def _parse_character_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse character evaluation results"""
        result = {}
        lines = response.split('\n')
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('CHARACTER_PRESENT:'):
                present_info = line.replace('CHARACTER_PRESENT:', '').strip().lower()
                result['character_present'] = 'yes' in present_info
            elif line.startswith('MOVEMENT_ANALYSIS:'):
                current_section = 'movement_analysis'
                result[current_section] = line.replace('MOVEMENT_ANALYSIS:', '').strip()
            elif line.startswith('SCORE:'):
                score_text = line.replace('SCORE:', '').strip()
                try:
                    result['score'] = int(score_text.split()[0])
                except:
                    result['score'] = 0
            elif line.startswith('JUSTIFICATION:'):
                current_section = 'justification'
                result[current_section] = line.replace('JUSTIFICATION:', '').strip()
            elif current_section and line and not line.startswith(('CHARACTER_', 'MOVEMENT_', 'SCORE:', 'JUST')):
                result[current_section] += ' ' + line
        
        return result


# Example usage
if __name__ == "__main__":
    # Initialize GeminiAPI
    gemini = GeminiAPI(
        api_keys=["YOUR_GEMINI_API_KEY", 
                 "YOUR_GEMINI_API_KEY", 
                 "YOUR_GEMINI_API_KEY"],
        proxy="YOUR_PROXY_URL"
    )
    
    # Initialize evaluator
    evaluator = PhysicalPlausibility(gemini)
    
    # Example evaluation
    video_path = "your_video_path_here"
    prompt = "your_prompt_here"
    
    results = evaluator.score_physical_plausibility(video_path, prompt)
    
    print(f"Final Score: {results['final_score']}/5")
    print(f"Description: {results['score_description']}")
    print(f"Character Count: {results['evaluation_results']['character_count']}")
    for i, char_eval in enumerate(results['evaluation_results']['character_evaluations']):
        print(f"Character {i+1} ({char_eval['character']}): {char_eval['score']}/5")
        print(f"  Analysis: {char_eval['justification']}")
