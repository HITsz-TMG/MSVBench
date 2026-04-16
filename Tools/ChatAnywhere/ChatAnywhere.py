import os
import base64
import mimetypes
from typing import List, Optional

from openai import OpenAI


class ChatAnywhereClient:
    """OpenAI-compatible client wrapper for ChatAnywhere multimodal generation.

    This class provides four helper methods:
    - generate_from_text
    - generate_from_images
    - generate_from_videos
    - generate_multimodal

    It uses data URLs for local files and supports both Chat Completions and
    the Responses API when needed (e.g., for videos).
    """

    def __init__(self, api_key: str = "YOUR_API_KEY_HERE", base_url: str = "https://api.chatanywhere.tech/v1"):
        key = api_key
        if not key or key == "YOUR_API_KEY_HERE":
            raise ValueError("API key not provided. Set CHATANYWHERE_API_KEY or OPENAI_API_KEY or pass api_key explicitly.")
        self.client = OpenAI(api_key=key, base_url=base_url)

    # -----------------------
    # Internal helpers
    # -----------------------
    def _file_to_data_url(self, path: str, default_mime: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or default_mime
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _bytes_to_data_url(self, data: bytes, mime: str) -> str:
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _extract_text(self, response) -> str:
        """Extract text output from either Chat Completions or Responses API result."""
        # Chat Completions style
        if hasattr(response, "choices"):
            try:
                return response.choices[0].message.content
            except Exception:
                pass

        # Responses API convenience attribute
        text = getattr(response, "output_text", None)
        if text:
            return text

        # Generic extraction from Responses object
        try:
            outputs = getattr(response, "output", None)
            if outputs:
                contents = outputs[0].content
                for item in contents:
                    # The SDK typically exposes type and text fields
                    if getattr(item, "type", None) == "output_text":
                        txt = getattr(item, "text", None)
                        if txt:
                            # Some SDKs wrap with .value or similar
                            return getattr(txt, "value", txt)
                # Fallback
                return str(response)
        except Exception:
            pass

        # Last resort
        return str(response)

    # -----------------------
    # Public generation methods
    # -----------------------
    def generate_from_text(self, prompt: str, model: str = "gemini-2.5-flash") -> str:
        """Generate text from a text prompt using Chat Completions.

        Example:
            client = ChatAnywhereClient(api_key="YOUR_API_KEY_HERE")
            text = client.generate_from_text("Hello", model="gemini-2.5-flash")
            print(text)
        """
        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_text(completion)

    def generate_from_images(self, image_paths: List[str], prompt: str, model: str = "gemini-2.5-flash") -> str:
        """Generate from images + text using Chat Completions (image_url data URLs).

        image_paths: list of local image file paths. They will be converted to
        base64 data URLs and sent as image_url content entries.

        Example:
            client = ChatAnywhereClient(api_key="YOUR_API_KEY_HERE")
            out = client.generate_from_images([
                "path/to/your/image1.png",
                "path/to/your/image2.jpg"
            ], "Describe these images", model="gemini-2.5-flash")
            print(out)
        """
        content = [{"type": "text", "text": prompt}]
        for p in image_paths:
            data_url = self._file_to_data_url(p, "image/png")
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        return self._extract_text(completion)

    def generate_from_videos(self, video_paths: List[str], prompt: str, model: str = "gemini-2.5-flash") -> str:
        """Generate from videos + text using Chat Completions.

        video_paths: list of local video file paths. They will be converted to
        base64 data URLs and sent as input_video content entries.

        Example:
            client = ChatAnywhereClient(api_key="YOUR_API_KEY_HERE")
            out = client.generate_from_videos([
                "path/to/your/video.mp4"
            ], "Analyze the video content", model="gemini-2.5-flash")
            print(out)
        """
        content = [{"type": "text", "text": prompt}]
        for p in video_paths:
            content.append({"type": "image_url", "image_url": {"url": p}})

        #print(content)

        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        return self._extract_text(completion)

    def generate_multimodal(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        video_paths: Optional[List[str]] = None,
        video_files: Optional[List] = None,
        model: str = "gemini-2.5-flash",
    ) -> str:
        """Generate from text + optional images/videos using Chat Completions.

        - image_paths: local image paths (converted to data URLs).
        - video_paths: local video paths (converted to data URLs).
        - video_files: file-like objects or raw bytes for videos.

        Example:
            client = ChatAnywhereClient(api_key="YOUR_API_KEY_HERE")
            out = client.generate_multimodal(
                prompt="Analyze these inputs comprehensively",
                image_paths=["path/to/your/image.png"],
                video_paths=["path/to/your/video.mp4"],
                model="gemini-2.5-flash",
            )
            print(out)
        """
        content = [{"type": "text", "text": prompt}]

        # Images
        if image_paths:
            for p in image_paths:
                data_url = self._file_to_data_url(p, "image/png")
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        #print(content)

        # Videos from paths
        if video_paths:
            for p in video_paths:
                content.append({"type": "image_url", "image_url": {"url": p}})
        #print(content)

        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        return self._extract_text(completion)


if __name__ == "__main__":
    # Initialize the client (ensure you have set the API key)
    client = ChatAnywhereClient(api_key="YOUR_API_KEY_HERE")
    
    # # 1) Text-only example
    # prompt = "Hello"
    # try:
    #     print("[Text]", client.generate_from_text(prompt, model="gemini-2.5-flash"))
    # except Exception as e:
    #     print("Text example error:", e)

    # # 2) Image + text example (set valid paths before running)
    # prompt = "Describe these images"
    # image_paths = ["path/to/your/image1.png", "path/to/your/image2.png"]
    # # if all(os.path.exists(p) for p in image_paths):
    # #     try:
    # #         print("[Images]", client.generate_from_images(image_paths, prompt, model="gemini-2.5-flash"))
    # #     except Exception as e:
    # #         print("Images example error:", e)
    # # else:
    # #     print("Skip images example: please provide valid image_paths.")

    # 3) Video + text example (set valid paths before running)
    prompt = "Conclude the two videos briefly"
    video_paths = ["path/to/your/video1.mp4", "path/to/your/video2.mp4"]
    try:
        print("[Videos]", client.generate_from_videos(video_paths, prompt, model="gemini-2.5-flash"))
    except Exception as e:
        print("Videos example error:", e)

    # # 4) Multimodal unified example (set valid paths before running)
    # prompt = "Conclude all the input images and videos separately and briefly"
    # image_paths = ["path/to/your/image1.png", "path/to/your/image2.png"]
    # video_paths = ["path/to/your/video1.mp4", "path/to/your/video2.mp4"]
    # # try:
    # #     print(
    # #         "[Multimodal]",
    # #         client.generate_multimodal(
    # #             prompt=prompt,
    # #             image_paths=image_paths,
    # #             video_paths=video_paths,
    # #             model="gemini-2.5-flash",
    # #         ),
    # #     )
    # # except Exception as e:
    # #     print("Multimodal example error:", e)