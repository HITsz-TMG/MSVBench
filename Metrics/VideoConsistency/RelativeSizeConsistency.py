import sys
import os
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(BASE_DIR, "Tools"))
from gemini_api import GeminiAPI


@dataclass
class ConsistencyResult:
    """Data class to store consistency analysis results"""
    consistency_score: float  # 0-1, higher is more consistent
    max_variation: float
    most_inconsistent_pair: Tuple[str, str]
    pair_ratios: Dict[str, List[float]]  # pair -> list of ratios


class RelativeSizeConsistency:
    """Tool class to analyze relative size consistency of characters across videos"""
    
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api
    
    def _create_prompt(self, character_list: List[str]) -> str:
        """Create detailed prompt for analyzing character size ratios"""
        pairs = [f"{character_list[i]} to {character_list[j]}" 
                for i in range(len(character_list)) 
                for j in range(i + 1, len(character_list))]
        
        return f"""
        Analyze this video and determine the relative size ratios between these character pairs: {', '.join(pairs)}

        IMPORTANT ANALYSIS REQUIREMENTS:
        1. Examine the entire video carefully, not just single frames
        2. Consider multiple appearances of characters throughout the video
        3. Account for perspective effects (closer objects appear larger)
        4. Account for distance from camera (farther objects appear smaller)
        5. Consider natural size differences between character types
        6. Look for consistent size relationships across different scenes/shots
        7. Ignore temporary perspective distortions (extreme close-ups, etc.)
        8. Focus on the characters' actual relative sizes when they appear together

        For each character pair that appears together in the video:

        PAIR: [character1 to character2]
        RATIO: [precise numerical ratio - how many times larger character1 is than character2]
        EXPLANATION: [detailed explanation of size relationship, considering: natural size differences, perspective effects, distance from camera, consistency across scenes, artistic choices]
        CONFIDENCE: [0.0-1.0 based on clarity of size relationship and number of observations]

        RATIO CALCULATION GUIDELINES:
        - 1.0 means characters are the same size
        - 1.5 means character1 is 1.5 times larger than character2
        - 0.7 means character1 is 0.7 times the size of character2 (smaller)
        - Consider the characters' body sizes, not temporary positioning
        - Average the size relationship across multiple appearances
        - Account for natural proportions (e.g., adult vs child, different species)

        If characters don't appear together clearly enough to judge size:
        PAIR: [character1 to character2]
        RATIO: N/A
        EXPLANATION: Characters not visible together with sufficient clarity
        CONFIDENCE: 1.0

        Be precise with numerical ratios and provide detailed explanations for your assessments.
        """
    
    def _parse_response(self, response: str) -> Dict[str, float]:
        """Parse Gemini response to extract ratios"""
        ratios = {}
        blocks = response.split("PAIR:")
        
        for block in blocks[1:]:
            lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
            if len(lines) < 2:
                continue
                
            try:
                pair = lines[0].strip()
                ratio_line = next((line for line in lines if line.startswith("RATIO:")), "")
                
                if "N/A" not in ratio_line:
                    ratio_match = re.search(r'RATIO:\s*([\d.]+)', ratio_line)
                    if ratio_match:
                        ratios[pair] = float(ratio_match.group(1))
            except:
                continue
                
        return ratios
    
    def analyze_consistency(self, video_list: List[str], character_list: List[str]) -> ConsistencyResult:
        """Analyze relative size consistency across videos"""
        print(f"Analyzing {len(video_list)} videos for {len(character_list)} characters...")
        
        # Collect all ratios
        all_ratios = {}
        
        for video_path in video_list:
            print(f"Analyzing: {os.path.basename(video_path)}")
            try:
                video_file = self.gemini_api.upload_video(video_path)
                prompt = self._create_prompt(character_list)
                response = self.gemini_api.generate_from_video_file(video_file, prompt)
                
                # Print the response for debugging
                print(f"\nGemini response for {os.path.basename(video_path)}:")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
                ratios = self._parse_response(response)
                print(f"Parsed ratios: {ratios}")
                
                for pair, ratio in ratios.items():
                    normalized_pair = pair.replace(" to ", "_to_").lower()
                    if normalized_pair not in all_ratios:
                        all_ratios[normalized_pair] = []
                    all_ratios[normalized_pair].append(ratio)
                    
            except Exception as e:
                print(f"Error analyzing {video_path}: {e}")
        
        # Calculate consistency metrics
        total_deviation = 0.0
        valid_pairs = 0
        max_variation = 0.0
        most_inconsistent_pair = ("", "")
        
        for pair, ratios in all_ratios.items():
            if len(ratios) >= 2:
                max_ratio = max(ratios)
                min_ratio = min(ratios)
                variation = max_ratio - min_ratio
                
                if variation > max_variation:
                    max_variation = variation
                    most_inconsistent_pair = (video_list[0], video_list[-1])  # Simplified
                
                if min_ratio > 0:
                    ratio_percentage = (max_ratio / min_ratio) * 100
                    deviation = abs(ratio_percentage - 100) / 100
                    total_deviation += deviation
                    valid_pairs += 1
        
        consistency_score = max(0, 1 - (total_deviation / valid_pairs)) if valid_pairs > 0 else 1.0
        
        return ConsistencyResult(
            consistency_score=consistency_score,
            max_variation=max_variation,
            most_inconsistent_pair=most_inconsistent_pair,
            pair_ratios=all_ratios
        )
    
    def print_report(self, result: ConsistencyResult):
        """Print analysis report"""
        print("\n" + "="*50)
        print("RELATIVE SIZE CONSISTENCY REPORT")
        print("="*50)
        print(f"Consistency Score: {result.consistency_score:.3f}")
        print(f"Max Variation: {result.max_variation:.3f}")
        print(f"Most Inconsistent Pair: {result.most_inconsistent_pair}")
        
        print("\nCharacter Pair Ratios:")
        for pair, ratios in result.pair_ratios.items():
            if len(ratios) >= 2:
                print(f"  {pair}: {ratios} (variation: {max(ratios) - min(ratios):.3f})")


# Example usage
if __name__ == "__main__":
    gemini = GeminiAPI(
        api_keys=["YOUR_GEMINI_API_KEY"],
        proxy="YOUR_PROXY_URL"
    )
    
    analyzer = RelativeSizeConsistency(gemini)
    
    video_list = [
        "your_first_video_path_here",
        "your_second_video_path_here",
        "your_third_video_path_here"
    ]
    character_list = ["your_first_character_here", "your_second_character_here"]
    
    result = analyzer.analyze_consistency(video_list, character_list)  # ratio_percentage = (max_ratio / min_ratio) * 100; deviation = abs(ratio_percentage - 100) / 100
    analyzer.print_report(result)
