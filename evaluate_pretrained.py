import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
model = YOLO('yolov8n.pt')

# Define class names sesuai dengan dataset.yaml
class_names = {
    0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
    5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
    10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
    15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
}

def evaluate_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    
    # Perform prediction
    results = model(image)
    
    # Get the first result
    result = results[0]
    
    # Convert image to numpy array for visualization
    image_np = np.array(image)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 8))
    
    # Display the image
    ax.imshow(image_np)
    
    # Plot each detection
    for box in result.boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        
        # Get class and confidence
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Get class name
        class_name = class_names.get(class_id, f"Unknown ({class_id})")
        
        # Create rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1, y1-5, f'{class_name} {conf:.2f}', 
                color='red', fontsize=10, backgroundcolor='white')
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save the plot
    output_path = image_path.replace('.jpg', '_detected.jpg')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return output_path

# Test the function with some images
test_images = [
    'data/processed/images/000001.jpg',
    'data/processed/images/000002.jpg',
    'data/processed/images/000003.jpg'
]

for image_path in test_images:
    try:
        output_path = evaluate_image(image_path)
        print(f"Detection completed. Result saved to: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}") 