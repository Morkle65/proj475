import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path
from ultralytics import YOLO

from scenes_classifier import SceneClassifier

device = torch.device("cuda")

scene_model = YOLO("runs/trained_objects/weights/best.pt")
backbone = scene_model.model.model[:10]

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder("data/places365/train", transform=transform)
val_dataset = datasets.ImageFolder("data/places365/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
print(f"Classes: {len(train_dataset.classes)}")

def one_epoch(model, loader, optimizer, criterion, device):
    total_loss = 0
    correct = 0
    total = 0
    model.train()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = (correct / total) * 100

    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    total_loss = 0
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            
            total_loss += loss.item()
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
    avg_loss = total_loss / len(loader)
    accuracy = (correct / total) * 100
        
    return avg_loss, accuracy

def training_loop(epochs, train_loader, val_loader, device):
    model = SceneClassifier(backbone, num_classes=365).to(device)
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    save_dir = Path("runs/trained_scenes")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        start = time.time()
        
        train_loss, train_accuracy = one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        elapsed = time.time() - start
        
        print(f"Epoch {epoch+1:3d}/{epochs}, Time: {elapsed:.1f}s, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            saved = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_accuracy": best_val_accuracy
            }
            torch.save(obj=saved, f=save_dir/"best.pt")
            print(f"Saved model (Accuracy = {best_val_accuracy:2f}%)")

training_loop(epochs=100, train_loader=train_loader, val_loader=val_loader, device=device)