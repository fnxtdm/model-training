import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms, models
from utils.labelme_dataset import LabelMeDataset
from tools.load_labels import save_labels
from utils.config import device, model_path, json_dir, image_dir


# Function to calculate feature size of VGG16 dynamically
def get_feature_size(model, input_shape):
    with torch.no_grad():
        dummy_input = torch.rand(1, *input_shape)  # Create a dummy input with the given shape
        features = model.features(dummy_input)  # Pass through the feature extractor of VGG16
        return features.view(1, -1).size(1)  # Return the flattened feature size


# Function to calculate evaluation metrics using PyTorch
def calculate_metrics(preds, labels, num_classes):
    preds = preds.cpu()
    labels = labels.cpu()

    # Accuracy
    accuracy = (preds == labels).sum().item() / len(labels)

    # Precision, Recall, F1-score for each class
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1_score = torch.zeros(num_classes)

    for i in range(num_classes):
        true_positive = ((preds == i) & (labels == i)).sum().item()
        false_positive = ((preds == i) & (labels != i)).sum().item()
        false_negative = ((preds != i) & (labels == i)).sum().item()

        if true_positive + false_positive > 0:
            precision[i] = true_positive / (true_positive + false_positive)
        if true_positive + false_negative > 0:
            recall[i] = true_positive / (true_positive + false_negative)
        if precision[i] + recall[i] > 0:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    avg_precision = precision.mean().item()
    avg_recall = recall.mean().item()
    avg_f1_score = f1_score.mean().item()

    print(
        f"Accuracy: {accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-score: {avg_f1_score:.4f}")
    return accuracy, avg_precision, avg_recall, avg_f1_score


# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224, standard for VGG16
    transforms.ToTensor(),
])

# Create the dataset
labelme_dataset = LabelMeDataset(json_dir, image_dir, transform)

# Set specific image priority (optional)
# Example: custom_dataset.set_priority('image1.png', 10)

# Create DataLoader without weighted sampling
data_loader = DataLoader(labelme_dataset, batch_size=32, shuffle=True)

# Calculate the number of classes
num_classes = len(labelme_dataset.all_labels)
save_labels(labelme_dataset.all_labels)

# Initialize VGG16 model
model = models.vgg16(pretrained=True)

# Freeze feature extractor layers
for param in model.features.parameters():
    param.requires_grad = False

# Dynamically compute feature size
input_shape = (3, 224, 224)  # (Channels, Height, Width)
feature_size = get_feature_size(model, input_shape)

# Modify classifier layers
model.classifier[0] = nn.Linear(feature_size, 4096)
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, data_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training complete.")
    return model


# Evaluation function
def evaluate_model(model, data_loader, labelme_dataset, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.append(predicted)
            all_labels.append(labels)

            # Debugging line for checking predictions
            for i in range(len(labels)):
                actual_label = labels[i].item()
                predicted_label = predicted[i].item()
                print(
                    f'Actual: {labelme_dataset.all_labels[actual_label]}, Predicted: {labelme_dataset.all_labels[predicted_label]}')

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate and print metrics
    calculate_metrics(all_preds, all_labels, num_classes)


# Main execution
if __name__ == "__main__":
    # Train the model
    trained_model = train_model(model, data_loader, criterion, optimizer, num_epochs=5)

    # Evaluate the model
    evaluate_model(trained_model, data_loader, labelme_dataset, num_classes)

    # Save the trained model
    torch.save(trained_model.state_dict(), model_path)
