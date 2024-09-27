import os
import sys
import json
import glob
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import json_dir, output_img_dir, output_label_dir

# Create output directories
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

def collect_unique_labels(json_dir):
    """Collect all unique labels from LabelMe JSON files."""
    label_set = set()
    for json_file in glob.glob(os.path.join(json_dir, '*.json')):
        with open(json_file, 'r') as f:
            labelme_data = json.load(f)
            for shape in labelme_data['shapes']:
                if shape['shape_type'] == 'rectangle':  # Only consider rectangle annotations
                    label_set.add(shape['label'])
    return label_set

def convert_to_yolo_format(labelme_data, img_width, img_height, class_index):
    """Convert LabelMe annotations to YOLO format."""
    labels = []
    for shape in labelme_data['shapes']:
        if shape['shape_type'] == 'rectangle':  # Only process rectangle annotations
            label = shape['label']
            if label in class_index:  # Check if label exists in mapping
                label_id = class_index[label]
                points = shape['points']
                x_min = min(points[0][0], points[1][0])
                y_min = min(points[0][1], points[1][1])
                x_max = max(points[0][0], points[1][0])
                y_max = max(points[0][1], points[1][1])

                # Calculate YOLO format coordinates
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                labels.append(f"{label_id} {x_center} {y_center} {width} {height}")
            else:
                print(f"Label '{label}' not found in mapping. Skipping...")  # Debug info

    return labels

def main():
    # Collect unique labels
    unique_labels = collect_unique_labels(json_dir)

    # Create a mapping from label names to class indices
    class_index = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    # Print unique labels with their indices
    print("Unique labels with indices:")
    for label, index in class_index.items():
        print(f"{index}: {label}")

    # Iterate through all JSON files to process images and labels
    for json_file in glob.glob(os.path.join(json_dir, '*.json')):
        with open(json_file, 'r') as f:
            labelme_data = json.load(f)

        # Read the corresponding image
        img_path = json_file.replace('.json', '.jpg')
        if not os.path.exists(img_path):
            img_path = json_file.replace('.json', '.png')

        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}. Skipping...")
            continue

        img = Image.open(img_path)
        img_width, img_height = img.size

        # Convert to YOLO format using the class_index mapping
        yolo_labels = convert_to_yolo_format(labelme_data, img_width, img_height, class_index)

        # Save image and labels
        img.save(os.path.join(output_img_dir, os.path.basename(img_path)))

        if yolo_labels:
            with open(os.path.join(output_label_dir, os.path.basename(json_file).replace('.json', '.txt')),
                      'w') as label_file:
                label_file.write('\n'.join(yolo_labels))

    print("Data preparation complete!")

# Run the main function
if __name__ == "__main__":
    main()
