
import re
import torch
import numpy as np
import torch.nn as nn
from trains.checkpoint import save_checkpoint
import os
import pandas as pd
import matplotlib.pyplot as plt


def train_one_epoch(model, train_loader, optimizer, device, config, global_step):
    criterion = nn.CrossEntropyLoss()
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    use_amp = device == "cuda"

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()
        global_step += 1

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc, global_step

def evaluate(model, val_loader, device, config):
    criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)

            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()

    val_loss = total_loss / len(val_loader)
    val_acc = correct / total

    return val_loss, val_acc

def save_history(history, name):

    csv_path = f"train_progress/csv/{name}.csv"
    img_path = f"train_progress/image/{name}.jpeg"

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)


    # Save CSV
    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)

    # Save Graph
    plt.figure(figsize=(8, 5))

    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DPO Training Progress")
    plt.legend()
    plt.grid(True)

    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"CSV saved: {csv_path}")
    print(f"Graph saved: {img_path}")


def train(model,train_loader,val_loader,optimizer, device,config):
    global_step = 0
    best_val_acc = -float("inf")
    history = []
    for epoch in range(config.epochs):

        train_loss,train_acc,global_step= train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            global_step
        )

        val_loss,val_acc= evaluate(model, val_loader, device, config)
       
        # best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_checkpoint(
                "checkpoints/bert/best.pt",
                model,
                optimizer,
                epoch+1,
                global_step
            )

        history.append({
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_acc": train_acc,
                            "val_acc": val_acc
                        })


        print(f"Epoch {epoch+1} | Train loss {train_loss:.4f} | Val Acc {val_acc:.4f}")
    save_history(history, "bert_history")




