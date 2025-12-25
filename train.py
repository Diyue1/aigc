import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score
import numpy as np

from model_rswa import AIGCDetector

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAIN_DIR = "/data/ziqiang/yjz/dataset/Benchmark/newTrain/train"
VAL_DIR = "/data/ziqiang/yjz/dataset/Benchmark/newTrain/val"

PHYSICAL_BATCH_SIZE = 64
TARGET_BATCH_SIZE = 128
ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE

LR = 2e-4
EPOCHS = 20
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecursiveBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'0_real': 0, '1_fake': 1}

        if not os.path.exists(root_dir):
            raise RuntimeError(f"路径不存在: {root_dir}")

        for root, dirs, files in os.walk(root_dir):
            folder_name = os.path.basename(root)
            if folder_name in self.class_to_idx:
                label = self.class_to_idx[folder_name]
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')):
                        path = os.path.join(root, file)
                        self.samples.append((path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"未找到数据！请检查路径结构。")
        print(f"[Dataset] {root_dir}: {len(self.samples)} images loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Warning: Failed to load image {path}: {e}")
            fallback = torch.zeros((3, 256, 256))
            if self.transform:
                fallback = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(fallback)
            return fallback, label


def train_model():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"环境: {device_name} | 策略: Batch {PHYSICAL_BATCH_SIZE} x {ACCUM_STEPS} = {TARGET_BATCH_SIZE}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        train_ds = RecursiveBinaryDataset(TRAIN_DIR, transform=transform)
        val_ds = RecursiveBinaryDataset(VAL_DIR, transform=transform)
        print(f"训练集样本数: {len(train_ds)}")
        print(f"验证集样本数: {len(val_ds)}")
    except Exception as e:
        print(f"数据集错误: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = AIGCDetector(num_classes=2, embed_dim=96).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_acc = 0.0
    print(f"开始训练... (Total {EPOCHS} Epochs)")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        last_i = -1
        for i, (inputs, labels) in enumerate(pbar):
            last_i = i
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs, recon_loss = model(inputs)
                    cls_loss = criterion(outputs, labels)
                    loss = (cls_loss + recon_loss) / ACCUM_STEPS
            else:
                outputs, recon_loss = model(inputs)
                cls_loss = criterion(outputs, labels)
                loss = (cls_loss + recon_loss) / ACCUM_STEPS

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % ACCUM_STEPS == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUM_STEPS
            if i % 20 == 0:
                pbar.set_postfix(loss=loss.item() * ACCUM_STEPS)

        if last_i >= 0 and (last_i + 1) % ACCUM_STEPS != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        model.eval()
        all_targets = []
        all_probs = []
        val_samples_count = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                val_samples_count += inputs.size(0)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs, _ = model(inputs)
                else:
                    outputs, _ = model(inputs)

                probs = torch.softmax(outputs, dim=1)[:, 1]

                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().float().numpy())

        all_preds = [1 if p > 0.5 else 0 for p in all_probs]
        val_acc = accuracy_score(all_targets, all_preds) * 100
        val_ap = average_precision_score(all_targets, all_probs) * 100
        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1} Result | Loss: {avg_loss:.4f} | Acc.: {val_acc:.2f}% | A.P.: {val_ap:.2f}% | Val samples used: {val_samples_count}/{len(val_ds)}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_rswa_model.pth")
            print(f"新纪录！模型已保存")

    print("训练全部完成。")


if __name__ == "__main__":
    train_model()