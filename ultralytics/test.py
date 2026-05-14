import cv2
import time
import glob
import os
from ultralytics import YOLO
import torch

def test_single_model_fps(model_path, image_folder, device='cuda'):
    """
    Test FPS for a single YOLO model.
    """
    model = YOLO(model_path)
    model.to(device)
    
    image_paths = glob.glob(os.path.join(image_folder, "*"))
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [path for path in image_paths 
                  if os.path.splitext(path)[1].lower() in valid_extensions]
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Found {len(image_paths)} test images")
    
    # Warmup
    print("Warming up model...")
    dummy_image = torch.randn(1, 3, 640, 640).to(device)
    for _ in range(10):
        _ = model(dummy_image)
    
    total_time = 0
    total_frames = 0
    
    print("Starting FPS test...")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Average of 3 runs per image
        for i in range(3):
            start_time = time.time()
            results = model(image, verbose=False)
            end_time = time.time()
            
            inference_time = end_time - start_time
            total_time += inference_time
            total_frames += 1
    
    if total_frames == 0:
        return
    
    avg_time_per_frame = total_time / total_frames
    fps = 1 / avg_time_per_frame
    
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Total Frames: {total_frames}")
    print(f"Avg Time: {avg_time_per_frame:.4f}s")
    print(f"FPS: {fps:.2f}")
    print("="*50)

if __name__ == "__main__":
    MODEL_PATH = "ultralytics/citrus_project/citrus_yolo11_run/weights/best.pt"
    IMAGE_FOLDER = "ultralytics/photo"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    test_single_model_fps(MODEL_PATH, IMAGE_FOLDER, device)