import os
os.environ['HTTP_PROXY'] = 'YOUR_PROXY_URL'  # Set to empty string or remove if no proxy is needed
os.environ['HTTPS_PROXY'] = 'YOUR_PROXY_URL'  # Set to empty string or remove if no proxy is needed
import torch
import numpy as np
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from tools.ShareGPT4Video.captioner.SimpleSlidingCaptioner import SimpleSlidingCaptioner

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
CAPTIONER_MODEL_PATH = os.path.join(
    BASE_DIR, "StoryVideoAlignment", "tools", "ShareGPT4Video", "captioner", "weights"
)
KALM_MODEL_PATH = os.path.join(
    BASE_DIR, "StoryVideoAlignment", "tools", "KaLM-embedding-multilingual-mini-instruct-v2"
)


class StoryVideoConsistency:
    def __init__(
        self,
        model_type='gemini',
        model='text-embedding-004',
        api_key='YOUR_GEMINI_API_KEY',
        captioner_model_path=CAPTIONER_MODEL_PATH,
    ):
        """
        Initialize the StoryVideoConsistency class

        Args:
            model_type (str): The type of model to use ('gemini' or 'kalm')
            model (str): The embedding model to use (default: 'text-embedding-004' for gemini)
            api_key (str): The API key for Google Gemini (only needed for gemini)
            captioner_model_path (str): Path to the video captioner model
        """
        self.model_type = model_type
        self.model = model

        # Initialize video captioner
        self.captioner = SimpleSlidingCaptioner(model_path=captioner_model_path)

        if model_type == 'gemini':
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"API key configuration error: {e}")
                raise
        elif model_type == 'kalm':
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
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'gemini' or 'kalm'")

    def generate_embeddings_gemini(self, text):
        """
        Generate embeddings using Gemini model

        Args:
            text (str): The text to generate embeddings for

        Returns:
            list: The embedding vector
        """
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        return result.embeddings[0].values

    def generate_embeddings_kalm(self, text):
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

    def generate_embeddings(self, text):
        """
        Generate embeddings for the given text using the selected model

        Args:
            text (str): The text to generate embeddings for

        Returns:
            list/numpy.ndarray: The embedding vector
        """
        if self.model_type == 'gemini':
            return self.generate_embeddings_gemini(text)
        elif self.model_type == 'kalm':
            return self.generate_embeddings_kalm(text)

    def cosine_similarity(self, vec_a, vec_b):
        """Calculate cosine similarity between two NumPy vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    def generate_video_description(self, video_path):
        """
        Generate video description using SimpleSlidingCaptioner

        Args:
            video_path (str): Path to the video file

        Returns:
            str: Generated video description
        """
        print(f"Generating video description for: {video_path}")
        description = self.captioner(video_path)
        print(f"Generated video description: {description[:100]}...")
        return description

    def calculate_similarity(self, video_description, generation_prompt, verbose=True):
        """
        Calculate the semantic similarity between video description and generation prompt

        Args:
            video_description (str): The description of the video
            generation_prompt (str): The generation prompt text
            verbose (bool): Whether to print detailed information

        Returns:
            float: The cosine similarity score
        """
        # Generate embeddings
        prompt_embedding = self.generate_embeddings(generation_prompt)
        description_embedding = self.generate_embeddings(video_description)

        if verbose:
            print(f"Successfully generated a {len(prompt_embedding)}-D vector for the prompt.")
            print(
                f"Successfully generated a {len(description_embedding)}-D vector for the video description."
            )
            print("-" * 30)

        # Convert to numpy arrays and calculate similarity
        prompt_vec = np.array(prompt_embedding)
        description_vec = np.array(description_embedding)

        similarity_score = self.cosine_similarity(prompt_vec, description_vec)

        if verbose:
            print(f"Prompt: {generation_prompt[:80]}...")
            print(f"Video description: {video_description[:80]}...")
            print(
                f"\nSemantic similarity computed by {self.model_type.upper()} ({self.model if self.model_type == 'gemini' else 'KaLM'}) is: {similarity_score:.4f}"
            )

        return similarity_score

    def calculate_similarity_from_video(self, video_path, generation_prompt, verbose=True):
        """
        Calculate the semantic similarity between video and generation prompt

        Args:
            video_path (str): Path to the video file
            generation_prompt (str): The generation prompt text
            verbose (bool): Whether to print detailed information

        Returns:
            tuple: (similarity_score, video_description)
        """
        # Generate video description
        video_description = self.generate_video_description(video_path)

        # Calculate similarity
        similarity_score = self.calculate_similarity(video_description, generation_prompt, verbose)

        return similarity_score, video_description


# Example usage
if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"  # Replace with an actual video path.
    generation_prompt = "your prompt here"  # Replace with an actual prompt.

    # Initialize with Gemini model
    print("Using Gemini model:")
    consistency_checker_gemini = StoryVideoConsistency(model_type='gemini', model='text-embedding-004')
    similarity_score_gemini, video_description = consistency_checker_gemini.calculate_similarity_from_video(
        video_path, generation_prompt
    )

    print("\n" + "=" * 50 + "\n")

    # Initialize with KaLM model
    print("Using KaLM model:")
    consistency_checker_kalm = StoryVideoConsistency(model_type='kalm')
    similarity_score_kalm, _ = consistency_checker_kalm.calculate_similarity_from_video(
        video_path, generation_prompt
    )

    print(f"\nFinal result:")
    print(f"Gemini similarity: {similarity_score_gemini:.4f}")
    print(f"KaLM similarity: {similarity_score_kalm:.4f}")
