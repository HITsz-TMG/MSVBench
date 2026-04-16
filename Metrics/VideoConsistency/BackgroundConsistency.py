import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from PIL import Image
from dreamsim import dreamsim
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, "tools", "Step1X-Edit"))
from Step1XEdit import Step1XEditProcessor

class BackgroundConsistencyProcessor:
    def __init__(self, 
                 model_path=os.path.join(CURRENT_DIR, "tools", "Step1X-Edit", "weights", "Step1X-Edit"),
                 seed=42,
                 num_steps=28,
                 cfg_guidance=6.0,
                 size_level=1024,
                 offload=False,
                 quantized=False,
                 prompt="Remove all people(boy, girl, man, woman) and animals from the image",
                 output_dir=os.path.join(CURRENT_DIR, "tools", "tmp"),
                 result_path=os.path.join(CURRENT_DIR, "tools", "MuDI", "results_dreamsim.txt")):
        
        self.processor = Step1XEditProcessor(
            model_path=model_path,
            seed=seed,
            num_steps=num_steps,
            cfg_guidance=cfg_guidance,
            size_level=size_level,
            offload=offload,
            quantized=quantized
        )
        
        self.prompt = prompt
        self.output_dir = output_dir
        self.result_path = result_path
        
        # Initialize DreamSim model
        device = "cuda"
        self.model, self.preprocess = dreamsim(pretrained=True, device=device)
        self.device = device
    
    def process_images_and_calculate_similarity(self, image_path1, image_path2):
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define output paths
        output1_path = os.path.join(self.output_dir, "processed_image_1.jpg")
        output2_path = os.path.join(self.output_dir, "processed_image_2.jpg")
        
        # Process both images
        self.processor.process_image(
            image_path=image_path1,
            prompt=self.prompt,
            output_path=output1_path
        )
        
        self.processor.process_image(
            image_path=image_path2,
            prompt=self.prompt,
            output_path=output2_path
        )
        
        # Load and preprocess the processed images
        img1 = self.preprocess(Image.open(output1_path)).to(self.device)
        img2 = self.preprocess(Image.open(output2_path)).to(self.device)
        distance = self.model(img1, img2)
        
        print(f"DreamSim distance between processed images: {distance.item()}")
        
        # Save the result to the specified file
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        with open(self.result_path, 'w') as f:
            f.write(f"Distance: {distance.item()}\n")
        
        print(f"Result saved to {self.result_path}")
        
        return distance.item()

# Example usage
if __name__ == "__main__":
    processor = BackgroundConsistencyProcessor()
    
    image_path1 = "your_first_image_path_here"
    image_path2 = "your_second_image_path_here"
    
    distance = processor.process_images_and_calculate_similarity(image_path1, image_path2)
    print(distance)
