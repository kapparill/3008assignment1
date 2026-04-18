import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from visualization import TrainingVisualizer
import os

# consts
SEED = 0 # set seed for assignment
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # mps not working, so this is slow
NUM_CLASSES = 10
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
torch.manual_seed(SEED)

# data prep
def load_cifar10_data(batch_size=BATCH_SIZE):
    # cifar10 noramlisation
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # noramlisation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # download datasets or use existing
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        transform=train_transform, 
        download=True
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        transform=test_transform, 
        download=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0 # 2 slow??
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_loader, test_loader


# SOFTMAX REGRESSION:
class SoftmaxRegression(nn.Module):
    """Linear classifier with softmax activation"""
    
    def __init__(self, num_classes=10):
        super(SoftmaxRegression, self).__init__()
        # Flatten 32x32x3 images to 3072-dimensional vectors
        self.fc = nn.Linear(32 * 32 * 3, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# 2 layer CNN:
class CNN2Layer(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN2Layer, self).__init__()
        
        self.features = nn.Sequential(
            # Layer 1: Conv -> BatchNorm -> ReLU -> Pool
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # Reduced channels from 32 to 16 for speed
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16
            
            # Layer 2: Conv -> BatchNorm -> ReLU -> Pool
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16 -> 8x8
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3), # Lower dropout for a smaller model
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Helper to create a block of Conv + BatchNorm + ReLU
def conv_bn_relu(in_f, out_f):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_f),
        nn.ReLU(inplace=True)
    )

class CNN_8Layer(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_8Layer, self).__init__()
        # 8 Conv layers total, 2 per pool block
        self.features = nn.Sequential(
            conv_bn_relu(3, 16), conv_bn_relu(16, 16), nn.MaxPool2d(2, 2), # 16x16
            conv_bn_relu(16, 32), conv_bn_relu(32, 32), nn.MaxPool2d(2, 2), # 8x8
            conv_bn_relu(32, 32), conv_bn_relu(32, 32), nn.MaxPool2d(2, 2), # 4x4
            conv_bn_relu(32, 32), conv_bn_relu(32, 32), nn.MaxPool2d(2, 2)  # 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64), # Reduced from 512 to 64 for CPU speed
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class CNN_16Layer(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_16Layer, self).__init__()
        # 16 Conv layers total, 4 per pool block
        def layer_block(in_f, out_f):
            return nn.Sequential(
                conv_bn_relu(in_f, out_f), conv_bn_relu(out_f, out_f),
                conv_bn_relu(out_f, out_f), conv_bn_relu(out_f, out_f),
                nn.MaxPool2d(2, 2)
            )
        
        self.features = nn.Sequential(
            layer_block(3, 16), layer_block(16, 16),
            layer_block(16, 32), layer_block(32, 32)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN_32Layer(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_32Layer, self).__init__()
        # 32 Conv layers total, 8 per pool block
        def deep_block(in_f, out_f):
            layers = [conv_bn_relu(in_f, out_f)]
            for _ in range(7):
                layers.append(conv_bn_relu(out_f, out_f))
            layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            deep_block(3, 16), deep_block(16, 16),
            deep_block(16, 32), deep_block(32, 32)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

        
# training and eval functions

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(test_loader)}")
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    # training model

    # visualiser in custom library init
    viz = TrainingVisualizer()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model.__class__.__name__}_best.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model to {checkpoint_path}")
        print()

        # add the epoch results to the visualisation
        viz.add_epoch(epoch+1, train_loss, train_acc, test_loss, test_acc)

    viz.save_history()
    viz.print_summary()
    viz.plot_all()
    
    return best_accuracy


# =
# main
# =
if __name__ == '__main__':
    print(f"Using device: {DEVICE}\n")
    train_loader, test_loader = load_cifar10_data(BATCH_SIZE)
    
    model_list = [
        # SoftmaxRegression, 
        # SimpleCNN,   # 2-Layer
        CNN_8Layer, 
        CNN_16Layer, 
        CNN_32Layer
    ]
    
    for model_class in model_list:
        print("=" * 60)
        print(f"TRAINING: {model_class.__name__}")
        print("=" * 60)
        
        model = model_class(NUM_CLASSES).to(DEVICE)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        
        best_acc = train_model(model, train_loader, test_loader)
        print(f"Final Best Accuracy for {model_class.__name__}: {best_acc:.2f}%\n")