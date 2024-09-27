
def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

# Save all labels and class count to file
def save_labels(all_labels, file_path='all_labels.txt'):
    with open(file_path, 'w') as f:
        for label in all_labels:
            f.write(f"{label}\n")