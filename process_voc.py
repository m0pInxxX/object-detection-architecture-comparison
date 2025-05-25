import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
import yaml

def create_folder_structure():
    """Create the necessary folder structure"""
    folders = [
        'data/processed/images',
        'data/processed/labels',
        'data/processed/train',
        'data/processed/val'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def convert_coordinates(size, box):
    """Convert VOC coordinates to YOLO format"""
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

def process_annotation(xml_path, class_mapping):
    """Process VOC XML annotation file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    # Process each object
    out_lines = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue
            
        cls = obj.find('name').text
        if cls not in class_mapping:
            continue
            
        cls_id = class_mapping[cls]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        bb = convert_coordinates((w,h), b)
        out_lines.append(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
    
    return out_lines

def process_voc_dataset(voc_path='path/to/VOCdevkit/VOC2012'):
    """Process VOC dataset into YOLO format"""
    # Create folder structure
    create_folder_structure()
    
    # Define class mapping
    class_mapping = {
        'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
    }
    
    # Save class mapping to yaml file
    yaml_data = {
        'names': {i: name for name, i in class_mapping.items()},
        'nc': len(class_mapping),
        'path': 'data/processed',
        'train': 'train',
        'val': 'val'
    }
    
    with open('data/processed/dataset.yaml', 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    
    # Process training set
    with open(os.path.join(voc_path, 'ImageSets/Main/train.txt'), 'r') as f:
        train_files = f.read().strip().split()
    
    # Process validation set
    with open(os.path.join(voc_path, 'ImageSets/Main/val.txt'), 'r') as f:
        val_files = f.read().strip().split()
    
    def process_set(file_list, set_name):
        for filename in file_list:
            # Copy and convert image
            src_img = os.path.join(voc_path, 'JPEGImages', f'{filename}.jpg')
            dst_img = f'data/processed/images/{filename}.jpg'
            shutil.copy2(src_img, dst_img)
            
            # Process annotation
            xml_path = os.path.join(voc_path, 'Annotations', f'{filename}.xml')
            label_lines = process_annotation(xml_path, class_mapping)
            
            if label_lines:
                # Save label file
                with open(f'data/processed/labels/{filename}.txt', 'w') as f:
                    f.write('\n'.join(label_lines))
                
                # Create symlinks
                os.symlink(os.path.abspath(dst_img),
                          f'data/processed/{set_name}/{filename}.jpg')
                os.symlink(os.path.abspath(f'data/processed/labels/{filename}.txt'),
                          f'data/processed/{set_name}/{filename}.txt')
    
    process_set(train_files, 'train')
    process_set(val_files, 'val')

if __name__ == '__main__':
    process_voc_dataset() 