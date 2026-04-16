import os
os.environ['HTTP_PROXY'] = 'YOUR_PROXY_URL'  # Set to empty string or remove if no proxy is needed
os.environ['HTTPS_PROXY'] = 'YOUR_PROXY_URL'  # Set to empty string or remove if no proxy is needed
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tools.ShareGPT4Video.captioner.SimpleSlidingCaptioner import SimpleSlidingCaptioner

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
CAPTIONER_MODEL_PATH = os.path.join(
    BASE_DIR, "StoryVideoAlignment", "tools", "ShareGPT4Video", "captioner", "weights"
)
KALM_MODEL_PATH = os.path.join(
    BASE_DIR, "StoryVideoAlignment", "tools", "KaLM-embedding-multilingual-mini-instruct-v2"
)


class VideoPromptConsistency:
    def __init__(self, video_path: str, prompt: str, captioner_model_path=CAPTIONER_MODEL_PATH):
        """
        Initialize the VideoPromptConsistency class

        Args:
            video_path (str): Path to the video file
            prompt (str): The generation prompt text
            captioner_model_path (str): Path to the video captioner model
        """
        self.video_path = video_path
        self.prompt = prompt

        # Initialize video captioner
        self.captioner = SimpleSlidingCaptioner(model_path=captioner_model_path)

        # Initialize KaLM model
        try:
            self.kalm_model = SentenceTransformer(
                KALM_MODEL_PATH,
                trust_remote_code=True,
                truncate_dim=None,
                model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
            )
            self.kalm_model.max_seq_length = 512
        except Exception as e:
            print(f"KaLM model loading error: {e}")
            raise

    def generate_embeddings(self, text):
        """
        Generate embeddings using KaLM model

        Args:
            text (str): The text to generate embeddings for

        Returns:
            numpy.ndarray: The embedding vector
        """
        embeddings = self.kalm_model.encode(
            [text], normalize_embeddings=True, batch_size=1, show_progress_bar=False
        )
        return embeddings[0]

    def cosine_similarity(self, vec_a, vec_b):
        """Calculate cosine similarity between two NumPy vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    def generate_video_description(self):
        """
        Generate video description using SimpleSlidingCaptioner

        Returns:
            str: Generated video description
        """
        print(f"Generating video description for: {self.video_path}")
        description = self.captioner(self.video_path)
        print(f"Generated video description: {description[:100]}...")
        return description

    def calculate_consistency(self, verbose=True):
        """
        Calculate the semantic similarity between video and prompt

        Args:
            verbose (bool): Whether to print detailed information

        Returns:
            tuple: (similarity_score, video_description)
        """
        # Generate video description
        video_description = self.generate_video_description()

        # Generate embeddings
        prompt_embedding = self.generate_embeddings(self.prompt)
        description_embedding = self.generate_embeddings(video_description)

        if verbose:
            print(f"Successfully generated a {len(prompt_embedding)}-D vector for the prompt.")
            print(
                f"Successfully generated a {len(description_embedding)}-D vector for the video description."
            )
            print("-" * 30)

        # Calculate similarity
        similarity_score = self.cosine_similarity(prompt_embedding, description_embedding)

        if verbose:
            print(f"Prompt: {self.prompt[:80]}...")
            print(f"Video description: {video_description[:80]}...")
            print(f"\nSemantic similarity computed by KaLM is: {similarity_score:.4f}")

        return similarity_score, video_description


# Example usage
if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"  # Replace with an actual video path.
    prompt = "your prompt here"  # Replace with your actual prompt.

    # Initialize and calculate consistency
    consistency_checker = VideoPromptConsistency(video_path, prompt)
    similarity_score, video_description = consistency_checker.calculate_consistency()

    print(f"\nFinal result:")
    print(f"Similarity: {similarity_score:.4f}")
