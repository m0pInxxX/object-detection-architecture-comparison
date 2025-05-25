import os
import torch
import yaml
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision.models.detection import SSD300_VGG16_Weights, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn
import cv2
from torchvision.transforms import functional as F
from tqdm.auto import tqdm
import pandas as pd
from ultralytics import YOLO

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluasi model pre-trained pada dataset VOC')
parser.add_argument('--data', type=str, default='data/processed/dataset.yaml', help='Path ke file dataset.yaml')
parser.add_argument('--model', type=str, default='all', choices=['ssd', 'faster_rcnn', 'yolov8', 'all'], help='Model yang akan dievaluasi')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
parser.add_argument('--conf', type=float, default=0.5, help='Threshold confidence untuk deteksi')
parser.add_argument('--num_samples', type=int, default=100, help='Jumlah sampel untuk evaluasi (0 untuk semua)')
parser.add_argument('--result_dir', type=str, default='results/pretrained', help='Direktori untuk menyimpan hasil')
parser.add_argument('--yolov8_weights', type=str, default='yolov8s.pt', help='Path ke weights YOLOv8 (yolov8s.pt, yolov8m.pt, dll)')

args = parser.parse_args()

# Pastikan direktori hasil ada
os.makedirs(args.result_dir, exist_ok=True)

# Device
device = torch.device(args.device)
print(f"Menggunakan device: {device}")

# Load dataset config
def load_dataset_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Berhasil memuat konfigurasi dataset dari: {yaml_path}")
        return config
    except (FileNotFoundError, PermissionError):
        print(f"Tidak dapat menemukan file dataset: {yaml_path}")
        return None

dataset_config = load_dataset_config(args.data)
if dataset_config is None:
    print("Gagal memuat konfigurasi dataset. Mencoba path alternatif...")
    alt_paths = ["dataset.yaml", "data/dataset.yaml", "Tugas 2/data/processed/dataset.yaml"]
    for path in alt_paths:
        dataset_config = load_dataset_config(path)
        if dataset_config is not None:
            break
    if dataset_config is None:
        print("Tidak dapat menemukan file konfigurasi dataset.")
        exit(1)

# Get class names
class_names = dataset_config.get('names', {})
if isinstance(class_names, dict):
    class_names = [class_names[i] for i in range(len(class_names))]
print(f"Dataset memiliki {len(class_names)} kelas: {class_names}")

# Load YOLOv8 model
def load_yolov8_model(weights_path='yolov8s.pt'):
    try:
        print(f"\nMemuat model YOLOv8 pre-trained ({weights_path})...")
        model = YOLO(weights_path)
        return model
    except Exception as e:
        print(f"Gagal memuat model YOLOv8: {e}")
        print("Mencoba mengunduh model dari Ultralytics...")
        try:
            model = YOLO('yolov8s.pt')
            return model
        except Exception as e2:
            print(f"Gagal memuat model YOLOv8 dari Ultralytics: {e2}")
            print("Lanjut tanpa model YOLOv8")
            return None

# Load pretrained models
def load_pretrained_models(model_name):
    models = {}
    
    if model_name in ['ssd', 'all']:
        print("\nMemuat model SSD pre-trained...")
        ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        ssd_model.eval().to(device)
        models['ssd'] = ssd_model
    
    if model_name in ['faster_rcnn', 'all']:
        print("\nMemuat model Faster R-CNN pre-trained...")
        faster_rcnn_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        faster_rcnn_model.eval().to(device)
        models['faster_rcnn'] = faster_rcnn_model
    
    if model_name in ['yolov8', 'all']:
        yolo_model = load_yolov8_model(args.yolov8_weights)
        if yolo_model:
            models['yolov8'] = yolo_model
    
    return models

# Prepare validation dataset
def prepare_validation_dataset(root_dir='data/processed', split='val'):
    from torch.utils.data import Dataset
    import glob
    
    class VOCDataset(Dataset):
        def __init__(self, root_dir, split='val'):
            self.root_dir = root_dir
            self.split = split
            self.img_files = glob.glob(os.path.join(root_dir, split, "images", "*.jpg"))
            
            if len(self.img_files) == 0:
                print(f"Warning: No images found in {os.path.join(root_dir, split, 'images')}")
                # Try alternative paths
                alt_paths = [
                    os.path.join('Tugas 2', root_dir, split, "images"),
                    os.path.join('..', root_dir, split, "images")
                ]
                for path in alt_paths:
                    self.img_files = glob.glob(os.path.join(path, "*.jpg"))
                    if len(self.img_files) > 0:
                        print(f"Found images in alternative path: {path}")
                        break
            
            print(f"Ditemukan {len(self.img_files)} gambar di {os.path.join(root_dir, split, 'images')}")
        
        def __len__(self):
            return len(self.img_files)
        
        def __getitem__(self, idx):
            img_path = self.img_files[idx]
            img = Image.open(img_path).convert("RGB")
            
            # Get corresponding label file
            label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
            
            boxes = []
            labels = []
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:  # class, x, y, w, h
                            cls_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Convert YOLO format to xmin, ymin, xmax, ymax
                            img_width, img_height = img.size
                            xmin = (x_center - width/2) * img_width
                            ymin = (y_center - height/2) * img_height
                            xmax = (x_center + width/2) * img_width
                            ymax = (y_center + height/2) * img_height
                            
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(cls_id + 1)  # +1 karena 0 adalah background di torchvision
            
            # Create targets
            target = {}
            if boxes:
                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
                target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros(0, dtype=torch.int64)
            
            # Apply transformations
            img_tensor = F.to_tensor(img)
            
            return img_tensor, target, img_path
    
    val_dataset = VOCDataset(root_dir=root_dir, split=split)
    return val_dataset

# Tambahkan di bagian atas file
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
COCO_TO_VOC = {
    0: 14,   # person
    1: 1,    # bicycle
    2: 6,    # car
    3: 13,   # motorbike
    4: 0,    # airplane/aeroplane
    5: 5,    # bus
    6: 18,   # train
    8: 3,    # boat
    14: 2,   # bird
    15: 7,   # cat
    16: 11,  # dog
    17: 12,  # horse
    18: 16,  # sheep
    19: 9,   # cow
    # Tambahkan mapping lain jika perlu
}

# Modifikasi fungsi visualisasi

def visualize_detection(image_path, boxes, labels, scores, class_names, save_path=None, model_name=""):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box, label, score in zip(boxes, labels, scores):
        label = int(label)
        # Mapping otomatis COCO ke VOC jika model COCO
        if model_name in ["ssd", "faster_rcnn", "yolov8"]:
            voc_idx = COCO_TO_VOC.get(label, None)
            if voc_idx is not None:
                label_name = VOC_CLASSES[voc_idx]
            else:
                continue  # skip kelas yang tidak ada di VOC
        else:
            label_name = class_names[label]

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label_name}: {score:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    img_pil = Image.fromarray(image)
    if save_path:
        img_pil.save(save_path)
        print(f"Visualisasi disimpan di {save_path}")
    return image

# Measure inference time
def measure_inference_time(model, image_tensor, num_runs=10, model_type=''):
    if model_type == 'yolov8':
        # YOLOv8 memiliki format input yang berbeda
        img = F.to_pil_image(image_tensor)
        
        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = model(img)
        
        # Measure time
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(img)
            torch.cuda.synchronize() if device.type == 'cuda' else None
        
        end_time = time.time()
    else:
        # Warm up for torchvision models
        for _ in range(3):
            with torch.no_grad():
                _ = model([image_tensor.to(device)])
        
        # Measure time
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model([image_tensor.to(device)])
            torch.cuda.synchronize() if device.type == 'cuda' else None
        
        end_time = time.time()
    
    # Calculate average time
    avg_time = (end_time - start_time) * 1000 / num_runs  # ms
    
    return avg_time

# Calculate IoU
def calculate_iou(box_a, box_b):
    # Calculate intersection
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

# Pindahkan fungsi evaluate_on_image ke sini
def evaluate_on_image(model, image_tensor, confidence_threshold=0.5, model_type=''):
    if model_type == 'yolov8':
        # YOLOv8 mengharapkan input yang berbeda dan menghasilkan output yang berbeda
        img = F.to_pil_image(image_tensor)
        with torch.no_grad():
            results = model(img)[0]  # Ambil hasil pertama
        
        boxes = []
        scores = []
        labels = []
        
        for box in results.boxes:
            boxes.append(box.xyxy[0].cpu().numpy())  # xyxy format
            scores.append(box.conf.cpu().numpy())
            labels.append(box.cls.cpu().numpy())
            
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
            
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)
        
        mask = scores >= confidence_threshold
        return boxes[mask], scores[mask], labels[mask].astype(int)
    else:
        # Metode evaluasi biasa untuk model torchvision
        with torch.no_grad():
            predictions = model([image_tensor.to(device)])
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()
        mask = pred_scores >= confidence_threshold
        boxes = pred_boxes[mask]
        scores = pred_scores[mask]
        labels = pred_labels[mask].astype(int)
        return boxes, scores, labels

# Main evaluation function
def evaluate_models(models, val_dataset, num_samples=100, confidence_threshold=0.5):
    results = {}
    
    # If num_samples is 0, use all samples
    if num_samples == 0 or num_samples > len(val_dataset):
        num_samples = len(val_dataset)
    
    # Get sample indices
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluasi model {model_name}...")
        print(f"{'='*50}")
        
        model_results = {
            'inference_times': [],
            'detections': []
        }
        
        # Create directory for visualizations
        vis_dir = os.path.join(args.result_dir, model_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Evaluate on sample images
        for i, idx in enumerate(tqdm(indices, desc=f"Evaluasi {model_name}")):
            image_tensor, target, img_path = val_dataset[idx]
            
            # Measure inference time
            if i < 10:  # Only measure time for first 10 images
                inference_time = measure_inference_time(model, image_tensor, model_type=model_name)
                model_results['inference_times'].append(inference_time)
            
            # Evaluate on image
            boxes, scores, labels = evaluate_on_image(model, image_tensor, confidence_threshold, model_type=model_name)
            
            # Store detection results
            model_results['detections'].append({
                'img_path': img_path,
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
                'target': target
            })
            
            # Visualize a few detections
            if i < 5:  # Only visualize first 5 images
                save_path = os.path.join(vis_dir, f"sample_{i}.jpg")
                visualize_detection(img_path, boxes, labels, scores, class_names, save_path, model_name)
        
        # Calculate average inference time
        avg_inference_time = np.mean(model_results['inference_times']) if model_results['inference_times'] else 0
        print(f"Rata-rata waktu inferensi {model_name}: {avg_inference_time:.2f} ms")
        
        # Calculate mAP (simplified)
        # Note: This is a simplified mAP calculation
        precision, recall = calculate_precision_recall(model_results['detections'], class_names)
        
        print(f"Precision rata-rata {model_name}: {np.mean(precision):.4f}")
        print(f"Recall rata-rata {model_name}: {np.mean(recall):.4f}")
        
        # Store results
        results[model_name] = {
            'inference_time': avg_inference_time,
            'precision': precision,
            'recall': recall,
            'mAP': np.mean(precision)  # Simplified mAP
        }
    
    return results

# Calculate precision and recall (simplified)
def calculate_precision_recall(detections, class_names):
    num_classes = len(class_names)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    
    # For this simplified evaluation, we'll just calculate the average
    # precision and recall across all detections
    for detection in detections:
        target = detection['target']
        pred_boxes = detection['boxes']
        pred_labels = detection['labels']
        
        # Process each class
        for cls_id in range(num_classes):
            # Get predictions for this class
            mask = pred_labels == (cls_id + 1)  # Add 1 because torchvision models consider 0 as background
            pred_boxes_cls = pred_boxes[mask]
            
            # Get ground truth for this class
            gt_mask = target['labels'] == (cls_id + 1)
            gt_boxes_cls = target['boxes'][gt_mask].cpu().numpy() if len(target['boxes']) > 0 else np.array([])
            
            # No predictions and no ground truth: perfect precision
            if len(pred_boxes_cls) == 0 and len(gt_boxes_cls) == 0:
                precision[cls_id] = 1.0
                recall[cls_id] = 1.0
                continue
            
            # No predictions but there is ground truth: zero recall
            if len(pred_boxes_cls) == 0:
                precision[cls_id] = 0.0
                recall[cls_id] = 0.0
                continue
            
            # Predictions but no ground truth: zero precision
            if len(gt_boxes_cls) == 0:
                precision[cls_id] = 0.0
                recall[cls_id] = 1.0
                continue
            
            # Calculate IoU between each prediction and ground truth
            tp = 0
            for pred_box in pred_boxes_cls:
                best_iou = 0.0
                for gt_box in gt_boxes_cls:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                
                if best_iou >= 0.5:  # IoU threshold
                    tp += 1
            
            # Calculate precision and recall
            precision[cls_id] = tp / len(pred_boxes_cls) if len(pred_boxes_cls) > 0 else 0
            recall[cls_id] = tp / len(gt_boxes_cls) if len(gt_boxes_cls) > 0 else 0
    
    return precision, recall

# Visualize results
def visualize_results(results):
    models = list(results.keys())
    metrics = ['inference_time', 'mAP']
    
    # Create DataFrame
    data = {
        'Model': [],
        'mAP': [],
        'Precision': [],
        'Recall': [],
        'Inference Time (ms)': []
    }
    
    for model_name, result in results.items():
        data['Model'].append(model_name)
        data['mAP'].append(result['mAP'])
        data['Precision'].append(np.mean(result['precision']))
        data['Recall'].append(np.mean(result['recall']))
        data['Inference Time (ms)'].append(result['inference_time'])
    
    df = pd.DataFrame(data)
    print("\nPerbandingan Model:")
    print(df)
    
    # Save results
    df.to_csv(os.path.join(args.result_dir, 'model_comparison.csv'), index=False)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot mAP
    plt.subplot(2, 2, 1)
    plt.bar(data['Model'], data['mAP'])
    plt.title('mAP (Mean Average Precision)')
    plt.ylim(0, 1)
    
    # Plot Precision
    plt.subplot(2, 2, 2)
    plt.bar(data['Model'], data['Precision'])
    plt.title('Precision')
    plt.ylim(0, 1)
    
    # Plot Recall
    plt.subplot(2, 2, 3)
    plt.bar(data['Model'], data['Recall'])
    plt.title('Recall')
    plt.ylim(0, 1)
    
    # Plot Inference Time
    plt.subplot(2, 2, 4)
    plt.bar(data['Model'], data['Inference Time (ms)'])
    plt.title('Inference Time (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, 'model_comparison.png'))
    print(f"Visualisasi hasil disimpan di {os.path.join(args.result_dir, 'model_comparison.png')}")

# Main execution
if __name__ == "__main__":
    # Load pre-trained models
    models = load_pretrained_models(args.model)
    
    # Prepare validation dataset
    val_dataset = prepare_validation_dataset()
    
    # Evaluate models
    results = evaluate_models(models, val_dataset, args.num_samples, args.conf)
    
    # Visualize results
    visualize_results(results) 
