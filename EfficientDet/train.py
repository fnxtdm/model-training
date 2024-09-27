import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

from torchvision import transforms
from utils.labelme_dataset import LabelMeDataset
from tools.load_labels import save_labels
from utils.config import device, model_path, json_dir, image_dir

# Image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # EfficientDet uses 512x512 images for better performance
    transforms.ToTensor(),
])

# Create the dataset
labelme_dataset = LabelMeDataset(json_dir, image_dir, transform)

# Create DataLoader
data_loader = DataLoader(labelme_dataset, batch_size=8, shuffle=True)

# Calculate the number of classes
num_classes = len(labelme_dataset.all_labels)
save_labels(labelme_dataset.all_labels)


# Initialize EfficientDet model
def create_efficientdet_model(num_classes):
    config = get_efficientdet_config('tf_efficientdet_d0')  # You can use d0-d7 depending on model size
    model = EfficientDet(config, pretrained_backbone=True)

    # Replace head with correct number of classes
    model.class_net = HeadNet(config, num_outputs=num_classes)

    # Bench for training
    bench = DetBenchTrain(model, config)
    return bench.to(device)


# Initialize EfficientDet model
model = create_efficientdet_model(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

def train_model(model, data_loader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets in data_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, targets)  # Pass both images and targets (which includes bboxes)
            loss = outputs['loss']  # EfficientDet returns a dictionary with a 'loss' key

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training complete.")
    return model


# Evaluation function (adjusted for EfficientDet)
def evaluate_model(model, data_loader, labelme_dataset, num_classes):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs['detections']  # Get predicted bounding boxes and classes

            # Convert predictions and labels for metrics calculation
            for i in range(len(predictions)):
                predicted_class = predictions[i]['labels']
                actual_class = labels[i]

                all_preds.append(predicted_class)
                all_labels.append(actual_class)

                actual_label = actual_class.item()
                predicted_label = predicted_class.item()
                print(
                    f'Actual: {labelme_dataset.all_labels[actual_label]}, Predicted: {labelme_dataset.all_labels[predicted_label]}')

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate and print metrics
    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    print(f'Accuracy: {accuracy:.4f}')


# Main execution
if __name__ == "__main__":
    # Train the model
    trained_model = train_model(model, data_loader, optimizer, num_epochs=5)

    # Evaluate the model
    evaluate_model(trained_model, data_loader, labelme_dataset, num_classes)

    # Save the trained model
    torch.save(trained_model.state_dict(), model_path)
