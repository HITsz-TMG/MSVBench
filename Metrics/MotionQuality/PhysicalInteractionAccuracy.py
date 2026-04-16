import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.gemini_api import GeminiAPI
from typing import Dict, Any, List, Optional


class PhysicalInteractionAccuracy:
    def __init__(self, gemini_api: GeminiAPI):
        """
        Initialize PhysicalInteractionAccuracy evaluator
        
        Args:
            gemini_api: Instance of GeminiAPI for video analysis
        """
        self.gemini_api = gemini_api
        self.scoring_criteria = {
            0: "No interacting objects found in video",
            1: "Objects present but motion/deformation completely incorrect",
            2: "Objects present with mostly incorrect motion/deformation", 
            3: "Objects present with partially correct motion/deformation",
            4: "Objects present with mostly correct motion/deformation",
            5: "Objects present with completely correct motion/deformation"
        }
    
    def analyze_expected_interactions(self, prompt: str) -> Dict[str, Any]:
        """
        Step 1: Analyze prompt to identify the most important interacting object
        and its expected motion/deformation
        
        Args:
            prompt: Description of the expected scene/interaction
            
        Returns:
            Dictionary containing analysis results
        """
        analysis_prompt = f"""
        Analyze the following prompt and identify the MOST IMPORTANT physical interaction:
        
        Prompt: "{prompt}"
        
        Please identify:
        1. The main object that characters interact with (choose only ONE most important object)
        2. The expected motion or deformation of this object based on the interaction
        3. Key physical principle that should govern this interaction
        
        Format your response as:
        MAIN_OBJECT: [the single most important interacting object]
        EXPECTED_MOTION: [describe the expected motion/deformation of this object]
        PHYSICAL_PRINCIPLE: [key physical law or realistic behavior]
        
        Be specific about the type of motion (rotation, translation, deformation, etc.) and direction.
        Focus on the most critical interaction only.
        """
        
        response = self.gemini_api.generate_from_text(analysis_prompt)
        #print(f"Analysis response: {response}")
        return self._parse_interaction_analysis(response)

    
    def evaluate_video_interactions(self, video_path: str, expected_interactions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Evaluate if the main object exists in video and assess motion accuracy
        
        Args:
            video_path: Path to the video file
            expected_interactions: Results from step 1 analysis
            
        Returns:
            Dictionary containing evaluation results and score
        """
        if not expected_interactions:
            return {"score": 0, "reason": "No expected interactions to evaluate"}
        
        # Check if the main object is present and evaluate its motion
        evaluation_prompt = f"""
        Based on this analysis of expected interaction:
        
        Main Object: {expected_interactions.get('main_object', '')}
        Expected Motion: {expected_interactions.get('expected_motion', '')}
        Physical Principle: {expected_interactions.get('physical_principle', '')}
        
        Watch this video and evaluate:
        1. Is the main object "{expected_interactions.get('main_object', '')}" present in the video?
        2. If present, how accurately does its motion/deformation match the expected behavior?
        3. Does the interaction follow the physical principle described?
        
        Give a score from 0-5 based on these detailed criteria:
        
        0 = Main object not found in video
        - The specified object is completely absent from the video
        - Example: Looking for a "ball" but no ball appears in any frame
        
        1 = Object present but motion completely incorrect
        - Object exists but behaves in a way that violates basic physics
        - Motion direction is opposite to expectation
        - Object ignores fundamental forces (gravity, momentum, etc.)
        - Example: A ball thrown upward continues accelerating upward indefinitely, or a dropped object falls upward
        
        2 = Object present with mostly incorrect motion
        - Object shows some realistic behavior but major aspects are wrong
        - Motion direction may be correct but speed/acceleration is highly unrealistic
        - Some physical principles followed but others completely ignored
        - Example: A bouncing ball bounces correctly once but then behaves erratically, or a liquid pours but flows upward
        
        3 = Object present with partially correct motion
        - Object follows basic physical principles but with noticeable inaccuracies
        - Motion is generally in the right direction but with timing or magnitude issues
        - Some aspects realistic, others clearly wrong
        - Example: A ball bounces but loses too much/little energy, or fabric moves but doesn't drape naturally
        
        4 = Object present with mostly correct motion
        - Object behavior is largely realistic with minor physics violations
        - Motion follows expected patterns with small deviations
        - Overall convincing but some details feel off
        - Example: A ball bounces realistically but bounces slightly too high, or liquid flows correctly but viscosity seems wrong
        
        5 = Object present with completely correct motion
        - Object behavior is physically accurate and realistic
        - Motion perfectly matches expectations based on real-world physics
        - All aspects of the interaction appear natural and convincing
        - Example: A ball bounces with proper energy loss and trajectory, or fabric drapes and moves exactly as expected
        
        Format response as:
        OBJECT_PRESENT: [Yes/No]
        MOTION_ACCURACY: [detailed analysis of the object's motion with specific observations]
        SCORE: [0-5]
        JUSTIFICATION: [explanation for the score with reference to the criteria above]
        """
        
        try:
            response = self.gemini_api.generate_from_videos([video_path], evaluation_prompt)
            results = self._parse_motion_evaluation(response)
            
            return {
                "score": results.get('score', 0),
                "reason": results.get('justification', ''),
                "motion_accuracy": results.get('motion_accuracy', ''),
                "object_present": results.get('object_present', False)
            }
            
        except Exception as e:
            print(f"Error in evaluating video interactions: {e}")
            return {"score": 0, "reason": f"Evaluation error: {str(e)}"}

    def score_physical_interaction_accuracy(self, video_path: str, prompt: str) -> Dict[str, Any]:
        """
        Complete evaluation pipeline combining both steps
        
        Args:
            video_path: Path to the video file
            prompt: Description of expected scene/interaction
            
        Returns:
            Dictionary containing complete evaluation results
        """
        print("Step 1: Analyzing expected interactions from prompt...")
        expected_interactions = self.analyze_expected_interactions(prompt)
        print(f"Expected interactions: {expected_interactions}")
        
        print("Step 2: Evaluating video against expected interactions...")
        evaluation_results = self.evaluate_video_interactions(video_path, expected_interactions)
        print(f"Evaluation results: {evaluation_results}")
        
        score = evaluation_results.get('score', 0)
        
        return {
            "video_path": video_path,
            "prompt": prompt,
            "expected_interactions": expected_interactions,
            "evaluation_results": evaluation_results,
            "final_score": score,
            "score_description": self.scoring_criteria.get(score, "Unknown score"),
            "detailed_analysis": {
                "step1_analysis": expected_interactions,
                "step2_evaluation": evaluation_results
            }
        }
    
    def _parse_interaction_analysis(self, response: str) -> Dict[str, Any]:
        """Parse the interaction analysis response"""
        result = {}
        lines = response.split('\n')
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('MAIN_OBJECT:'):
                current_section = 'main_object'
                result[current_section] = line.replace('MAIN_OBJECT:', '').strip()
            elif line.startswith('EXPECTED_MOTION:'):
                current_section = 'expected_motion'
                result[current_section] = line.replace('EXPECTED_MOTION:', '').strip()
            elif line.startswith('PHYSICAL_PRINCIPLE:'):
                current_section = 'physical_principle'
                result[current_section] = line.replace('PHYSICAL_PRINCIPLE:', '').strip()
            elif current_section and line and not line.startswith(('MAIN_', 'EXPEC', 'PHYS')):
                result[current_section] += ' ' + line
        
        return result
    
    def _parse_detection_results(self, response: str) -> Dict[str, Any]:
        """Parse object detection results"""
        result = {}
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('OBJECT_PRESENT:'):
                object_info = line.replace('OBJECT_PRESENT:', '').strip().lower()
                result['object_present'] = 'yes' in object_info
            elif line.startswith('MOTION_ACCURACY:'):
                result['motion_accuracy'] = line.replace('MOTION_ACCURACY:', '').strip()
        
        return result
    
    def _parse_motion_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse motion evaluation results"""
        result = {}
        lines = response.split('\n')
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('OBJECT_PRESENT:'):
                object_info = line.replace('OBJECT_PRESENT:', '').strip().lower()
                result['object_present'] = 'yes' in object_info
            elif line.startswith('MOTION_ACCURACY:'):
                current_section = 'motion_accuracy'
                result[current_section] = line.replace('MOTION_ACCURACY:', '').strip()
            elif line.startswith('SCORE:'):
                score_text = line.replace('SCORE:', '').strip()
                try:
                    result['score'] = int(score_text.split()[0])
                except:
                    result['score'] = 0
            elif line.startswith('JUSTIFICATION:'):
                current_section = 'justification'
                result[current_section] = line.replace('JUSTIFICATION:', '').strip()
            elif current_section and line and not line.startswith(('OBJECT_', 'MOTION_', 'SCORE:', 'JUST')):
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
    evaluator = PhysicalInteractionAccuracy(gemini)
    
    # Example evaluation
    video_path = "your_video_path_here"
    prompt = "your_prompt_here"
    
    results = evaluator.score_physical_interaction_accuracy(video_path, prompt)
    
    print(f"Final Score: {results['final_score']}/5")
    print(f"Description: {results['score_description']}")
    print(f"Analysis: {results['evaluation_results']['reason']}")
