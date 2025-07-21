import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from models.load_cifar10 import load_cifar10
from models.resnet import resnet50
from torch.utils.data import random_split

def train_with_tensorboard(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    log_dir="runs/exp1",
    save_dir="results"
):
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    best_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # --- TensorBoard Logging ---
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        print(f"Epoch {epoch+1}/{num_epochs} | TrainLoss: {train_loss:.4f} Acc: {train_acc:.4f} | ValLoss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # --- Model Saving ---
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))

    writer.close()
    print("Training finished!")

    # Optionally return the training history
    return history

NET=resnet50()
BATCH_SIZE=256
DEVICE='cuda'if torch.cuda.is_available() else 'cpu'
LOSS=nn.CrossEntropyLoss()
lr=0.001
OPTIMIZER=torch.optim.SGD(NET.parameters(),lr=lr,weight_decay=0.00001,momentum=0.9)
NUM_EPOCHS=100
dataset=load_cifar10(True)
test_iter=load_cifar10(False)

num_train = int(len(dataset) * 0.8)
num_val = len(dataset) - num_train
train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

train_with_tensorboard(NET,train_iter,val_iter,LOSS,OPTIMIZER,DEVICE,NUM_EPOCHS)