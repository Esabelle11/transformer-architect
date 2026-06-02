
import re
import torch
import numpy as np
import torch.nn as nn
from trains.checkpoint import save_checkpoint


def train_one_epoch(model,train_loader,optimizer,device,config,global_step):
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0

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

        total_loss += loss.item()
        global_step += 1

    return total_loss / len(train_loader),global_step


def evaluate(model, val_loader, device, config):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total





def train(model,train_loader,val_loader,optimizer, device,config):
    global_step = 0
    best_val_acc = -float("inf")
    for epoch in range(config.epochs):

        train_loss, global_step= train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            global_step
        )

        val_acc= evaluate(model, val_loader, device, config)
       
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

        print(f"Epoch {epoch+1} | Train loss {train_loss:.4f} | Val Acc {val_acc:.4f}")




