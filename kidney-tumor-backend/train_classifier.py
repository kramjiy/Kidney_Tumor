import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models.classifier import get_classifier
from pathlib import Path

# Config
DATA_DIR = Path("data/classifier_dataset")
SAVE_PATH = Path("data/models/classifier.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

# Transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_ds = ImageFolder(DATA_DIR / "train", transform=transform)
val_ds   = ImageFolder(DATA_DIR / "val", transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = get_classifier("resnet50", num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss_sum:.4f}")

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print("✅ Classifier trained & saved")
