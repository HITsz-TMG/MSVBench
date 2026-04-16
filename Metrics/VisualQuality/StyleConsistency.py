import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
CSD_CODE_DIR = os.path.join(BASE_DIR, "Benchmark", "vistorybench", "vistorybench", "bench", "style", "csd")
CSD_MODEL_PATH = os.path.join(BASE_DIR, "VisualQuality", "checkpoints", "CSD-ViT-L", "pytorch_model.bin")

sys.path.append(CSD_CODE_DIR)
from csd_model import CSD_CLIP
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path

class StyleConsistencyEvaluator:
    def __init__(self):
        self.csd_encoder = self._load_csd()
    
    def _load_csd(self):
        csd_image_encoder = CSD_CLIP(only_global_token=True)
        state_dict = torch.load(CSD_MODEL_PATH, map_location="cpu")
        csd_image_encoder.load_state_dict(state_dict, strict=False)
        
        # Freeze all parameters
        csd_image_encoder.eval()
        for param in csd_image_encoder.parameters():
            param.requires_grad = False
            
        # Ensure float32 is used
        csd_image_encoder = csd_image_encoder.to(dtype=torch.float32).cuda()
        for module in csd_image_encoder.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.to(dtype=torch.float32)
        
        return csd_image_encoder
    
    def _extract_frames_1fps(self, video_path):
        """Extract one frame per second from a video."""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample one frame per second
            if frame_count % fps == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Return a preprocessed tensor."""
        image_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_array).cuda()[None,...].permute(0,3,1,2)
        return tensor
    
    def _force_resize_to_224(self, image_tensor):
        """Resize to 224x224."""
        return T.functional.resize(
            image_tensor,
            size=[224, 224], 
            interpolation=T.InterpolationMode.BILINEAR
        )
    
    def _encode(self, image_tensor):
        """Encode image into style features."""
        preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                        (0.26862954, 0.26130258, 0.27577711)),
        ])
        input_image_tensor = preprocess(image_tensor).to(device=image_tensor.device, dtype=image_tensor.dtype)
        image_embeds = self.csd_encoder(input_image_tensor)['style']
        return image_embeds
    
    def _calculate_similarity(self, embed1, embed2):
        """Compute cosine similarity between two feature vectors."""
        cos_sim = F.cosine_similarity(embed1, embed2, dim=-1)
        return cos_sim.mean()
    
    def evaluate(self, video_path1, video_path2=None):
        """
        Compute style consistency
        Args:
            video_path1: Path to the first video
            video_path2: Path to the second video (optional)
        Returns:
            float: Average style consistency score
        """
        if video_path2 is None:
            # Compute internal style consistency for a single video
            return self._evaluate_internal_consistency(video_path1)
        else:
            # Compute style consistency between two videos
            return self._evaluate_cross_consistency(video_path1, video_path2)
    
    def _evaluate_internal_consistency(self, video_path):
        """Compute average style consistency among frames in one video."""
        frames = self._extract_frames_1fps(video_path)
        
        if len(frames) < 2:
            print(f"Insufficient video frames for internal consistency computation: {len(frames)}")
            return 0.0
        
        similarities = []
        
        # Compare all frame pairs
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                tensor1 = self._preprocess(frames[i])
                tensor2 = self._preprocess(frames[j])
                
                tensor1 = self._force_resize_to_224(tensor1)
                tensor2 = self._force_resize_to_224(tensor2)
                
                embed1 = self._encode(tensor1)
                embed2 = self._encode(tensor2)
                
                sim = self._calculate_similarity(embed1, embed2)
                similarities.append(sim.item())
        
        avg_similarity = np.mean(similarities)
        print(f"Internal style consistency - total frames: {len(frames)}, comparisons: {len(similarities)}, average similarity: {avg_similarity:.4f}")
        
        return avg_similarity
    
    def _evaluate_cross_consistency(self, video_path1, video_path2):
        """Compute average style consistency between two videos."""
        frames1 = self._extract_frames_1fps(video_path1)
        frames2 = self._extract_frames_1fps(video_path2)
        
        if len(frames1) == 0 or len(frames2) == 0:
            print(f"Video frame extraction failed: video1={len(frames1)}, video2={len(frames2)}")
            return 0.0
        
        similarities = []
        
        # Compare each frame with all frames in the other video
        for i in range(len(frames1)):
            for j in range(len(frames2)):
                tensor1 = self._preprocess(frames1[i])
                tensor2 = self._preprocess(frames2[j])
                
                tensor1 = self._force_resize_to_224(tensor1)
                tensor2 = self._force_resize_to_224(tensor2)
                
                embed1 = self._encode(tensor1)
                embed2 = self._encode(tensor2)
                
                sim = self._calculate_similarity(embed1, embed2)
                similarities.append(sim.item())
        
        avg_similarity = np.mean(similarities)
        print(f"Cross-video style consistency - video1 frames: {len(frames1)}, video2 frames: {len(frames2)}, comparisons: {len(similarities)}, average similarity: {avg_similarity:.4f}")
        
        return avg_similarity

# Keep legacy functions for backward compatibility
def load_csd():
    csd_image_encoder = CSD_CLIP(only_global_token=True)
    state_dict = torch.load(CSD_MODEL_PATH, map_location="cpu")
    csd_image_encoder.load_state_dict(state_dict, strict=False)
    
    # Freeze all parameters
    csd_image_encoder.eval()
    for param in csd_image_encoder.parameters():
        param.requires_grad = False
        
    # Ensure float32 is used
    csd_image_encoder = csd_image_encoder.to(dtype=torch.float32)
    for module in csd_image_encoder.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.to(dtype=torch.float32)
    
    return csd_image_encoder

def encode(image_tensor,csd_encoder):
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                    (0.26862954, 0.26130258, 0.27577711)),
    ])
    input_image_tensor = preprocess(image_tensor).to(device=image_tensor.device, dtype=image_tensor.dtype)
    image_embeds = csd_encoder(input_image_tensor)['style']
    return image_embeds

def preprocess(image: Image.Image) -> torch.Tensor:
    """Return a preprocessed tensor with gradients enabled."""
    image_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_array).cuda()[None,...].permute(0,3,1,2)
    return tensor.requires_grad_(True)  # Important: enable gradients


def force_resize_to_224(image_tensor):
    """Force-resize any input to 224x224."""
    return T.functional.resize(
        image_tensor,  # Automatically converts to tensor and adds batch dimension
        size=[224, 224], 
        interpolation=T.InterpolationMode.BILINEAR
    )

def get_csd_loss(style_img = None, render_img = None):
    # Initialize model
    csd_encoder = load_csd().cuda()

    # # Load and preprocess images
    # if style_img is None:
    #     style_img = Image.open('style_image.jpg').convert('RGB')
    # if render_img is None:
    #     render_img = Image.open('render_image.jpg').convert('RGB')

    # Style image does not require gradients; rendered image does
    if isinstance(style_img, (Image.Image)):
        print(f'Style image input is PIL image; converting to tensor next')
        style_tensor = preprocess(style_img).detach()  # detach blocks gradients
    elif isinstance(style_img, (torch.Tensor)):
        style_tensor = style_img

    if isinstance(render_img, (Image.Image)):
        print(f'Render image input is PIL image; converting to tensor next')
        render_tensor = preprocess(render_img)         # keep gradients
    elif isinstance(render_img, (torch.Tensor)):
        render_tensor = render_img

    style_tensor = force_resize_to_224(style_tensor) 

    print(f'style_tensor:{style_tensor.shape}')
    print(f'render_tensor:{render_tensor.shape}')

    # Forward pass
    style_embed = encode(style_tensor, csd_encoder)
    render_embed = encode(render_tensor, csd_encoder)

    print(f'style_embed:{style_embed.shape}')
    print(f'render_embed:{render_embed.shape}')

    # Compute cosine-similarity loss
    cos_sim = F.cosine_similarity(style_embed, render_embed, dim=-1)
    loss = 1 - cos_sim.mean()

    # Backpropagation test
    # loss.backward()

    print(f'Initial loss value: {loss.item():.4f}')
    print(f'Render image gradient exists: {render_tensor.grad is not None}')
    if render_tensor.grad is not None:
        print(f'Gradient norm: {render_tensor.grad.norm().item():.4f}')

    return loss



def get_csd_score(csd_encoder = None, img1_path = None, img2_path = None):
    # Initialize model
    if csd_encoder is None:
        csd_encoder = load_csd().cuda()

    # # Load and preprocess images
    # if img1 is None:
    #     img1 = Image.open('style_image.jpg').convert('RGB')
    # if img2 is None:
    #     img2 = Image.open('render_image.jpg').convert('RGB')

    Img1 = Image.open(img1_path).convert('RGB')
    image1_tensor = preprocess(Img1).detach() # detach blocks gradients

    Img2 = Image.open(img2_path).convert('RGB')
    image2_tensor = preprocess(Img2).detach() # keep gradients

    image1_tensor = force_resize_to_224(image1_tensor) 
    image2_tensor = force_resize_to_224(image2_tensor) 

    print(f'image1_tensor:{image1_tensor.shape}')
    print(f'image2_tensor:{image2_tensor.shape}')

    # Forward pass
    image1_embed = encode(image1_tensor, csd_encoder)
    image2_embed = encode(image2_tensor, csd_encoder)

    print(f'image1_embed:{image1_embed.shape}')
    print(f'image2_embed:{image2_embed.shape}')

    # Compute cosine-similarity loss
    cos_sim = F.cosine_similarity(image1_embed, image2_embed, dim=-1)
    cos_sim_mean = cos_sim.mean()

    return cos_sim, cos_sim_mean

### MSVBench SC Eval ###
if __name__ == "__main__":
    evaluator = StyleConsistencyEvaluator()
    
    # Example: compute internal style consistency for a single video.
    video_path = "path/to/your/video.mp4"  # Replace with an actual video path.
    internal_score = evaluator.evaluate(video_path)
    print(f'Internal style consistency score: {internal_score:.4f}')
    
    # Example: compute style consistency between two videos.
    video_path1 = "path/to/your/video1.mp4"  # Replace with an actual video path.
    video_path2 = "path/to/your/video2.mp4"  # Replace with an actual video path.
    cross_score = evaluator.evaluate(video_path1, video_path2)
    print(f'Cross-video style consistency score: {cross_score:.4f}')