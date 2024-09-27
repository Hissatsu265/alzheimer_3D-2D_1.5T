import torch.optim as optim
import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src.src_data4.dataloadder import MRIDataset
from src.src_data4.model import AlzheimerNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def save_images_with_predictions(model, test_loader, output_dir):
    model.eval()  # Set the model to evaluation mode
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    with torch.no_grad():  # Disable gradient computation for testing
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Save each image with its predicted label
            for j in range(inputs.size(0)):
                img = transforms.ToPILImage()(inputs.cpu()[j])
                predicted_label = predicted[j].item()
                true_label = labels[j].item()

                img.save(os.path.join(output_dir, f"img_{i}_{j}_pred_{predicted_label}_true_{true_label}.png"))

def train_model(model, train_loader,test_loader, criterion, optimizer, num_epochs=25):
    model.train()  # Set the model to training mode
    train_loss_history = []
    train_acc_history = []
    precision_history = []
    recall_history = []
    f1_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_preds = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear gradients
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        # Calculate precision, recall, and f1 for this epoch
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        precision_history.append(precision)
        recall_history.append(recall)
        f1_history.append(f1)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    print('Finished Training')
    plot_metrics(train_acc_history, precision_history, recall_history, f1_history)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def plot_metrics(acc, precision, recall, f1):
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc, label='Accuracy')
    plt.plot(epochs, precision, label='Precision')
    plt.plot(epochs, recall, label='Recall')
    plt.plot(epochs, f1, label='F1 Score')

    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Training Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig("training_metrics.png")
    plt.show()
# Load the dataset
dataset = MRIDataset(root_dir="/home/jupyter-iec_iot13_toanlm/data/data4/data_2D", transform=transform)

# Split dataset into train and test (80% train, 20% test)
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
# //=======================Load======================================================
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Initialize the model and optimizer
model = AlzheimerNet(num_classes=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model)  # Use multiple GPUs if available
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader,test_loader, criterion, optimizer, num_epochs=25)

# Save the model weights
torch.save(model.state_dict(), "alzheimer_model_data4.pth")

# After training, save predictions on test set images to the specified output directory
output_dir = "/home/jupyter-iec_iot13_toanlm/result/result_data4"
# save_images_with_predictions(model, test_loader, output_dir)





