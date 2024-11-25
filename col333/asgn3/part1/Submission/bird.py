import csv
import sys
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

torch.manual_seed(0)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class birdClassifier(nn.Module):
    def __init__(self):
        super(birdClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Initial convolution
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Layer 1
            ResNetBlock(64, 64),
            ResNetBlock(64, 64),

            # Layer 2
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 128),

            # Layer 3
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 256),

            # Layer 4
            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 512),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),  # Adjust the output layer for 8 classes
        )

    def forward(self, x):
        return self.model(x)

def train(model: birdClassifier, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Adam, scheduler: torch.optim.lr_scheduler.StepLR, device: torch.device, num_epochs: int=15) -> None:
    for _ in range(num_epochs):
        model.train()
        
        for images, labels in train_loader:
            images, labels = images.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        scheduler.step()

if __name__ == "__main__": 
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "bird.pth"

# -------------------------------------- Train ------------------------------------
if trainStatus == "train":
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    STEP_SIZE = 5
    GAMMA = 0.1
    
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4838, 0.493, 0.4104], std=[0.1924, 0.1919, 0.1931])
    ])
    
    # Load the dataset
    root_dir = dataPath
    train_dataset = ImageFolder(root_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model, loss, and optimizer
    model = birdClassifier().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # Training loop
    train(model, train_loader, criterion, optimizer, scheduler, device)
    
    # Saving the model as .pth
    torch.save(model.state_dict(), modelPath)
# ----------------------------------- Inference ------------------------------------
else:
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = birdClassifier().to(device)
    
    model_path = modelPath
    model.load_state_dict(torch.load(model_path))
    
    # Data augmentation and normalization for training
    test_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4838, 0.493, 0.4104], std=[0.1924, 0.1919, 0.1931])
    ])
    
    # Load the dataset
    root_dir = dataPath
    test_dataset = ImageFolder(root_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Inference loop
    model.eval()
    total = 0
    results = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device).float()
            outputs = model(images)
            results.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            total += 1
    
    with open("bird.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label'])
        for label in results:
            writer.writerow([label])
    
    
print(f"Training: {trainStatus}")
print(f"path to dataset: {dataPath}")
print(f"path to model: {modelPath}")