import os
import time
import requests
import PIL.Image
from io import BytesIO
from google import genai
from google.genai import types
from typing import List, Optional, Union, Dict, Any

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_TEST_VIDEO = "path/to/your/test_video.mp4"


class GeminiAPI:
    def __init__(self, api_keys: Union[str, List[str]], proxy: Optional[str] = None):
        # Convert single API key to list if needed
        self.api_keys = [api_keys] if isinstance(api_keys, str) else api_keys
        self.current_key_index = 0
        self.max_retries = 10
        
        # Add video file cache to avoid repeated uploads
        # Cache structure: {api_key_index: {video_path: video_file_object}}
        self.uploaded_videos = {i: {} for i in range(len(self.api_keys))}
        # Track ownership of uploaded file objects by API key index
        self.file_owner_by_name: Dict[str, int] = {}
        
        # Set up proxy if provided
        if proxy:
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
        
        # Initialize client with first API key
        self.client = genai.Client(api_key=self.api_keys[0])

    def is_current_key_usable(self, test_video_path: Optional[str] = None, prompt: str = "Summarize the video.", model: str = "gemini-2.5-flash") -> bool:
        """Check if the current API key is usable by calling generate_from_videos.

        - Only tests the current key without retries or switching; temporarily sets max_retries to 1 to avoid switching.
        - Returns True if non-empty text is generated; returns False if the call fails or returns empty text.
        """
        # Default test video
        video_path = test_video_path or DEFAULT_TEST_VIDEO

        if not os.path.exists(video_path):
            print(f"[Key Check] Test video does not exist: {video_path}")
            return False

        # Store and set minimal retries to avoid key switching
        original_max_retries = self.max_retries
        self.max_retries = 1
        try:
            text = self.generate_from_videos([video_path], prompt, model=model)
            usable = isinstance(text, str) and len(text.strip()) > 0
            if usable:
                print("[Key Check] Current key is usable: received non-empty text response.")
            else:
                print("[Key Check] Current key is not usable: returned empty text.")
            return usable
        except Exception as e:
            print(f"[Key Check] Current key failed to call generate_from_videos: {e}")
            return False
        finally:
            self.max_retries = original_max_retries

    @staticmethod
    def is_key_usable(api_key: str, test_video_path: Optional[str] = None, prompt: str = "Summarize the video.", proxy: Optional[str] = None, model: str = "gemini-2.5-flash") -> bool:
        """Check whether a GIVEN api_key is usable via a one-shot video analysis.

        Creates a temporary client and performs a single request without retries.
        Returns True on success (non-empty text), False otherwise.
        """
        # Temporarily set proxy env if provided and track previous values
        prev_http = os.environ.get("HTTP_PROXY")
        prev_https = os.environ.get("HTTPS_PROXY")
        try:
            if proxy:
                os.environ["HTTP_PROXY"] = proxy
                os.environ["HTTPS_PROXY"] = proxy

            # Use GeminiAPI to create an isolated client for the given key
            checker = GeminiAPI(api_keys=[api_key], proxy=proxy)
            return checker.is_current_key_usable(test_video_path=test_video_path, prompt=prompt, model=model)
        finally:
            # Restore previous proxy env to avoid side effects
            if proxy is not None:
                if prev_http is None:
                    os.environ.pop("HTTP_PROXY", None)
                else:
                    os.environ["HTTP_PROXY"] = prev_http
                if prev_https is None:
                    os.environ.pop("HTTPS_PROXY", None)
                else:
                    os.environ["HTTPS_PROXY"] = prev_https
    
    def _switch_api_key(self):
        """Switch to the next available API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        print(f"Switched to API key index {self.current_key_index}")
        print(f"Note: Videos will need to be re-uploaded for the new API key")
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic"""
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    raise Exception(f"Failed after {self.max_retries} attempts. Last error: {str(e)}")
                
                print(f"Attempt {retries} failed: {str(e)}. Waiting 10s before retrying...")
                time.sleep(10)
                self._switch_api_key()
        
    def generate_from_text(self, prompt: str, model: str = "gemini-2.5-flash") -> str:
        """Generate content from text-only input"""
        def _generate():
            response = self.client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        
        return self._execute_with_retry(_generate)
        
    def generate_from_images(self, image_paths: List[str], prompt: str, model: str = "gemini-2.5-flash") -> str:
        """Generate content from text and images"""
        def _generate():
            # Load all images
            pil_images = [PIL.Image.open(path) for path in image_paths]
            
            # Create content array starting with the prompt
            contents = [prompt] + pil_images
            
            # Send request to API
            response = self.client.models.generate_content(
                model=model,
                contents=contents
            )
            
            return response.text
        
        return self._execute_with_retry(_generate)
    
    def upload_video(self, video_path: str):
        """Upload a video and cache the file object"""
        current_cache = self.uploaded_videos[self.current_key_index]
        
        if video_path in current_cache:
            print(f"Video already uploaded for current API key: {video_path}")
            return current_cache[video_path]
        
        def _upload():
            print(f"Uploading video: {video_path}...")
            video_file = self.client.files.upload(file=video_path)
            print(f"Completed upload: {video_file.uri}")
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(1)
                video_file = self.client.files.get(name=video_file.name)
            
            if video_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed for {video_path}")
            
            # Cache the uploaded video for current API key
            current_cache[video_path] = video_file
            # Record ownership for validation when switching keys
            self.file_owner_by_name[video_file.name] = self.current_key_index
            return video_file
        
        return self._execute_with_retry(_upload)
    
    def generate_from_video_file(self, video_file, prompt: str, model: str = "gemini-2.5-flash") -> str:
        """Generate content from pre-uploaded video file"""
        def _generate():
            # Ensure the provided file belongs to the current API key
            self._ensure_video_belongs_to_current_key(video_file)
            response = self.client.models.generate_content(
                model=model,
                contents=[video_file, prompt]
            )
            return response.text
        
        return self._execute_with_retry(_generate)
    
    def generate_from_videos(self, video_paths: List[str], prompt: str, model: str = "gemini-2.5-flash") -> str:
        """Generate content from text and videos"""
        def _generate():
            uploaded_videos = []
            contents = []
            current_cache = self.uploaded_videos[self.current_key_index]

            # Load all videos (use cache if available)
            for video_path in video_paths:
                if video_path in current_cache:
                    video_file = current_cache[video_path]
                    print(f"Using cached video for current API key: {video_path}")
                else:
                    print(f"Uploading video: {video_path}...")
                    video_file = self.client.files.upload(file=video_path)
                    print(f"Completed upload: {video_file.uri}")
                    
                    # Wait for processing
                    while video_file.state.name == "PROCESSING":
                        print('.', end='')
                        time.sleep(1)
                        video_file = self.client.files.get(name=video_file.name)
                    
                    if video_file.state.name == "FAILED":
                        raise ValueError(f"Video processing failed for {video_path}")
                    
                    # Cache the uploaded video for current API key
                    current_cache[video_path] = video_file
                
                uploaded_videos.append(video_file)
                contents.append(video_file)
            
            # End with prompt
            contents.append(prompt)
            
            # Generate content using all videos for the current key
            response = self.client.models.generate_content(
                model=model,
                contents=contents
            )
            
            return response.text
        
        return self._execute_with_retry(_generate)
    
    def generate_multimodal(self, prompt: str, image_paths: Optional[List[str]] = None, video_paths: Optional[List[str]] = None, video_files: Optional[List] = None, model: str = "gemini-2.5-flash") -> str:
        """Generate content from a mix of text, images, and videos"""
        def _generate():
            # Start with prompt
            contents = [prompt]
            
            # Add images if provided
            if image_paths:
                for img_path in image_paths:
                    contents.append(PIL.Image.open(img_path))
            
            # Add videos if provided
            uploaded_videos = []
            if video_paths:
                current_cache = self.uploaded_videos[self.current_key_index]
                for video_path in video_paths:
                    if video_path in current_cache:
                        video_file = current_cache[video_path]
                        print(f"Using cached video for current API key: {video_path}")
                    else:
                        print(f"Uploading video: {video_path}...")
                        video_file = self.client.files.upload(file=video_path)
                        print(f"Completed upload: {video_file.uri}")
                        
                        # Wait for processing
                        while video_file.state.name == "PROCESSING":
                            print('.', end='')
                            time.sleep(1)
                            video_file = self.client.files.get(name=video_file.name)
                        
                        if video_file.state.name == "FAILED":
                            raise ValueError(f"Video processing failed for {video_path}")
                        
                        # Cache the uploaded video for current API key
                        current_cache[video_path] = video_file
                    
                    uploaded_videos.append(video_file)
                    contents.append(video_file)
            
            # Add pre-uploaded video files if provided
            if video_files:
                for video_file in video_files:
                    self._ensure_video_belongs_to_current_key(video_file)
                    contents.append(video_file)
                    
            # Generate content with all inputs
            response = self.client.models.generate_content(
                model=model,
                contents=contents
            )
            
            return response.text
        
        return self._execute_with_retry(_generate)

    def _ensure_video_belongs_to_current_key(self, video_file):
        """Validate that a given uploaded file was created under the current API key."""
        file_name = getattr(video_file, "name", None)
        if not file_name:
            raise ValueError("Unknown video file object without name; cannot validate ownership.")
        owner_index = self.file_owner_by_name.get(file_name)
        if owner_index is None:
            raise ValueError("Video file ownership unknown; upload within this client before use.")
        if owner_index != self.current_key_index:
            raise ValueError(
                "Video file was uploaded under a different API key; re-upload required for current key."
            )
    
    def images_text_to_image(self, prompt: str, image_paths: List[str], output_path: Optional[str] = None, model: str = "gemini-2.5-flash-exp-image-generation") -> PIL.Image.Image:
        """Generate an image from input images and text prompt"""
        def _generate():
            # Load all images
            pil_images = [PIL.Image.open(path) for path in image_paths]
            
            # Create content array with prompt and images
            contents = [prompt] + pil_images
            
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            
            # Extract image from response
            image = None
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(f"Response text: {part.text}")
                elif part.inline_data is not None:
                    image = PIL.Image.open(BytesIO(part.inline_data.data))
                    
                    # Save image if output path is provided
                    if output_path:
                        image.save(output_path)
                        print(f"Image saved to: {output_path}")
            
            if image is None:
                raise ValueError("No image generated in response")
                
            return image
        
        return self._execute_with_retry(_generate)


# Example usage:
if __name__ == "__main__":
    # Initialize the API with multiple keys and proxy
    api_keys = ["YOUR_API_KEY_HERE"]
    proxy = "socks5://YOUR_PROXY_HERE"
    gemini = GeminiAPI(
        api_keys=api_keys,
        proxy=proxy
    )

    # # Example: Process text only
    # prompt = "Explain quantum computing in simple terms."
    # text_result = gemini.generate_from_text(prompt)
    # print("Text-only Result:")
    # print(text_result)

    # # Example: Process images
    # prompt = "What do these images have in common?"
    # image_paths = ["path/to/your/image1.png", "path/to/your/image2.png"]
    # image_result = gemini.generate_from_images(image_paths, prompt)
    # print("Image Analysis Result:")
    # print(image_result)

    # Example: Process videos
    prompt = "Summarize the video."
    video_paths = ["path/to/your/video.mp4"]  # 可以替换为你具体的 DEFAULT_TEST_VIDEO 路径
    video_result = gemini.generate_from_videos(video_paths, prompt)
    print("Video Analysis Result:")
    print(video_result)

    # # Example: Multimodal input
    # prompt = "Explain the content of these two images and two videos."
    # image_paths = ["path/to/your/image1.png", "path/to/your/image2.png"]
    # video_paths = ["path/to/your/video1.mp4", "path/to/your/video2.mp4"]
    # multimodal_result = gemini.generate_multimodal(
    #     prompt=prompt,
    #     image_paths=image_paths,
    #     video_paths=video_paths
    # )
    # print("Multimodal Analysis Result:")
    # print(multimodal_result)

    # # Example: Process audios
    # prompt = "Speech to text."
    # audio_paths = ["path/to/your/audio.wav"]
    # audio_result = gemini.generate_from_videos(audio_paths, prompt)
    # print("Audio Analysis Result:")
    # print(audio_result)

    # # Example: Generate image from text and reference images
    # prompt = 'aspect_ratio="16:9", image generation: The little boy in image 1 met the fox in image 2 in the rose garden in image 3, and they exchanged warm greetings'
    # image_paths = ["path/to/your/character1.jpg", "path/to/your/character2.jpg", "path/to/your/environment.png"]
    # output_path = "path/to/your/generated_image_with_reference.png"
    # generated_image = gemini.images_text_to_image(
    #     prompt=prompt,
    #     image_paths=image_paths,
    #     output_path=output_path
    # )