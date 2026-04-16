import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(BASE_DIR, "Tools"))
from gemini_api import GeminiAPI
from typing import Tuple, Dict, Any
import re

class ClothesColorConsistency:
    def __init__(self, api_keys, proxy=None):
        """Initialize the consistency checker with Gemini API"""
        self.gemini = GeminiAPI(api_keys=api_keys, proxy=proxy)
        
    def check_consistency(self, image_path1: str, image_path2: str, character_prompt: str) -> Dict[str, Any]:
        """
        Check character existence and clothing/shoes consistency between two images
        
        Args:
            image_path1: Path to first image
            image_path2: Path to second image
            character_prompt: Description of the character to look for (e.g., "a little girl in white")
            
        Returns:
            Dictionary containing scores and detailed analysis
        """
        
        prompt = f"""
        Please analyze these two images based on the character description: "{character_prompt}"

        This is a two-step analysis:

        **STEP 1: Character Existence Check**
        First, determine if the described character exists in BOTH images.
        - If the character exists in both images: Continue to Step 2
        - If the character is missing from either image: Give 0 points for final score and stop analysis

        **STEP 2: Clothing and Shoes Consistency Analysis**
        If the character exists in both images, analyze the consistency of their clothing and shoes.

        **IMPORTANT: Consider perspective and viewing angle changes**
        - The same clothing item may appear different from front vs back view
        - Different camera angles can change how colors, patterns, and details appear
        - Lighting conditions may affect color perception
        - Focus on identifying whether it's the SAME clothing item despite these variations

        Scoring criteria for Step 2 (1-5 points):
        - 5 points: Completely identical - Same clothing item, colors and styles match perfectly (accounting for perspective)
        - 4 points: Very similar - Same clothing item, colors and styles are basically the same, minor differences due to angle/lighting/perspective
        - 3 points: Quite similar - Likely the same clothing item, but some uncertainty due to perspective or visible differences
        - 2 points: Partially similar - Could be the same item but significant differences make it unclear
        - 1 point: Basically different - Different clothing items, though similar type
        - 0 points: Character missing from either image

        Please analyze in detail:
        1. Character existence: Does the described character appear in both images?
        2. Clothing analysis: Compare the character's clothing considering possible perspective changes (front/back/side views)
        3. Shoes analysis: Compare the character's shoes considering possible viewing angle differences
        4. Perspective considerations: How might different viewing angles affect the appearance of the same items?
        5. Overall consistency: Final judgment accounting for perspective variations

        Please output in the following format:
        Character Existence: [Yes/No - exists in both images]
        Final Score: X points (0 if character missing from either image, 1-5 if character exists in both)
        Clothing Analysis: [Detailed comparison of clothing features, considering perspective changes]
        Shoes Analysis: [Detailed comparison of shoes features, considering viewing angles]
        Overall Judgment: [Final reasoning for the consistency score, accounting for perspective variations]
        """
        
        try:
            # Call Gemini API to analyze the images
            response = self.gemini.generate_from_images([image_path1, image_path2], prompt)
            
            # Parse the response to extract score and analysis
            result = self._parse_response(response)
            result['raw_response'] = response
            result['character_prompt'] = character_prompt
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'character_existence': None,
                'final_score': None,
                'clothing_analysis': None,
                'shoes_analysis': None,
                'overall_judgment': None,
                'character_prompt': character_prompt,
                'raw_response': None
            }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the Gemini API response to extract structured information"""
        
        # Extract character existence
        existence_match = re.search(r'Character Existence:\s*(.+?)(?=Final Score:|$)', response, re.DOTALL)
        character_existence = existence_match.group(1).strip() if existence_match else None
        
        # Extract final score
        score_match = re.search(r'Final Score:\s*(\d+)\s*points?', response)
        final_score = int(score_match.group(1)) if score_match else None
        
        # Extract different analysis sections
        clothing_match = re.search(r'Clothing Analysis:\s*(.+?)(?=Shoes Analysis:|$)', response, re.DOTALL)
        shoes_match = re.search(r'Shoes Analysis:\s*(.+?)(?=Overall Judgment:|$)', response, re.DOTALL)
        judgment_match = re.search(r'Overall Judgment:\s*(.+?)$', response, re.DOTALL)
        
        return {
            'character_existence': character_existence,
            'final_score': final_score,
            'clothing_analysis': clothing_match.group(1).strip() if clothing_match else None,
            'shoes_analysis': shoes_match.group(1).strip() if shoes_match else None,
            'overall_judgment': judgment_match.group(1).strip() if judgment_match else None
        }
    
    def batch_check_consistency(self, image_pairs: list, character_prompt: str) -> list:
        """
        Check consistency for multiple image pairs with the same character prompt
        
        Args:
            image_pairs: List of tuples containing (image_path1, image_path2)
            character_prompt: Description of the character to look for
            
        Returns:
            List of consistency check results
        """
        results = []
        for i, (img1, img2) in enumerate(image_pairs):
            print(f"Processing pair {i+1}/{len(image_pairs)}: {img1} vs {img2}")
            result = self.check_consistency(img1, img2, character_prompt)
            result['image_path1'] = img1
            result['image_path2'] = img2
            results.append(result)
        
        return results


def main():
    """Main function to test the consistency checker"""
    
    # Initialize the API with your keys and proxy
    api_keys = [
        "YOUR_GEMINI_API_KEY", 
        "YOUR_GEMINI_API_KEY", 
        "YOUR_GEMINI_API_KEY"
    ]
    proxy = "YOUR_PROXY_URL"
    
    # Create consistency checker
    checker = ClothesColorConsistency(api_keys=api_keys, proxy=proxy)
    
    # Test with the provided image paths and character prompt
    image_path1 = "your_first_image_path_here"
    image_path2 = "your_second_image_path_here"
    character_prompt = "your_character_prompt_here"
    
    print("Analyzing character and clothing consistency...")
    print(f"Character: {character_prompt}")
    print(f"Image 1: {image_path1}")
    print(f"Image 2: {image_path2}")
    print("=" * 50)
    
    # Perform consistency check
    result = checker.check_consistency(image_path1, image_path2, character_prompt)
    
    # Display results
    if 'error' in result and result['error']:
        print(f"Error: {result['error']}")
    else:
        print(f"Character Existence: {result['character_existence']}")
        print(f"Final Consistency Score: {result['final_score']}/5")
        print("\nClothing Analysis:")
        print(result['clothing_analysis'])
        print("\nShoes Analysis:")
        print(result['shoes_analysis'])
        print("\nOverall Judgment:")
        print(result['overall_judgment'])
        
        # Also print raw response for debugging
        print("\n" + "="*50)
        print("Raw Response:")
        print(result['raw_response'])


if __name__ == "__main__":
    main()
