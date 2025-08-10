# Kutuphaneler
import os, time, warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# AMP uyumlulugu (otomatik secim)
try:
    from torch.amp import autocast, GradScaler
    scaler = GradScaler()
    print("AMP: torch.amp kullaniliyor")
except:
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    print("AMP: torch.cuda.amp kullaniliyor")

# Config
data_dir = os.path.join(".", "archive", "Brain_Cancer raw MRI data", "Brain_Cancer")
num_epochs = 50
batch_size = 64
learning_rate = 0.001
weight_decay = 1e-4
model_save_path = "./trained_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nDevice: {device}\n")

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

print(f"Toplam veri: {len(dataset)}, Egitim: {len(train_dataset)}, Dogrulama: {len(val_dataset)}")
print(f"Siniflar: {dataset.classes}\n")

# Model
class DeepCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = DeepCNN(num_classes=len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

train_acc_list, val_acc_list = [], []

# Egitim dongusu
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        progress.set_postfix({
            "Loss": f"{train_loss:.3f}",
            "Accuracy": f"{100 * train_correct / train_total:.2f}%"
        })

    scheduler.step()
    train_acc = 100 * train_correct / train_total
    train_acc_list.append(train_acc)
    print(f"Egitim Tamamlandi - Dogruluk: {train_acc:.2f}")

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    val_acc_list.append(val_acc)
    print(f"Dogrulama - Loss: {val_loss:.3f}, Dogruluk: {val_acc:.2f}%\n")

# Model kaydi
torch.save(model.state_dict(), model_save_path)
print(f"Model kaydedildi: {model_save_path}\n")

# Dogruluk grafigi
plt.figure(figsize=(10, 5))
plt.plot(train_acc_list, label="Train Accuracy", marker='o')
plt.plot(val_acc_list, label="Val Accuracy", marker='x')
plt.title("Egitim vs Dogrulama Dogruluk")
plt.xlabel("Epoch")
plt.ylabel("Dogruluk (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Ornek tahminler ve confusion matrix
model.eval()
true_all, pred_all = [], []
num_show, shown = 10, 0
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.ravel()

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        true_all.extend(labels.cpu().numpy())
        pred_all.extend(predicted.cpu().numpy())

        for i in range(images.size(0)):
            if shown >= num_show:
                break
            img = images[i].cpu() * 0.5 + 0.5
            axs[shown].imshow(img.permute(1, 2, 0).numpy())
            axs[shown].set_title(f"Pred: {dataset.classes[predicted[i]]}\nTrue: {dataset.classes[labels[i]]}")
            axs[shown].axis('off')
            shown += 1
        if shown >= num_show:
            break

plt.tight_layout()
plt.suptitle("Ornek Tahminler", fontsize=16, y=1.05)
plt.show()

cm = confusion_matrix(true_all, pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# Final degerlendirme
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

final_acc = 100 * test_correct / test_total
print(f"Final Dogruluk: {final_acc:.2f}%")
print(f"Final Kayip: {test_loss / len(val_loader):.4f}")