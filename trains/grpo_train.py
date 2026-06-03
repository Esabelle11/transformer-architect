import re
from typing import Any
import torch
import numpy as np
from torch.nn import functional as F
from trains.checkpoint import save_checkpoint
import copy
import os
import pandas as pd
import matplotlib.pyplot as plt


@torch.no_grad()
def sample(model, x, tokenizer, max_new_tokens=64, temperature=0.8):

    model.eval()

    B, T = x.shape
    max_len = T + max_new_tokens

    out = torch.full(
        (B, max_len),
        tokenizer.pad_token_id,
        device=x.device,
        dtype=x.dtype
    )

    out[:, :T] = x

    cur_len = T

    for _ in range(max_new_tokens):
        logits = model(out[:, :cur_len])

        logits = logits[:, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, 1)

        out[:, cur_len] = next_token.squeeze(-1)

        cur_len += 1

        if (next_token == tokenizer.eos_token_id).all():
            break

    return out[:, :cur_len]

def extract_number(text):
    import re
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None


def reward_fn(pred, gt):
    try:
        # print("pred: ",pred)
        # print("gt: ",gt)
        p = extract_number(pred)
        g = extract_number(gt)

        # print("p: ",p)
        # print("g: ",g)

        if p is None or g is None:
            return -1.0

        p = float(p)
        g = float(g)

        error = abs(p - g)
        if error < 1e-6:
            return 1.0
        else:
            error = min(error, 100)
            return -error * 0.1
            # return -math.log1p(error)

        # return 1.0 if abs(p - g) < 1e-6 else -abs(p - g) * 0.01

    except:
        return -1.0

def compute_logprob(model, x, tokenizer):
    logits = model(x[:, :-1])
    log_probs = torch.log_softmax(logits, dim=-1)

    target = x[:, 1:].unsqueeze(-1)

    token_logprob = log_probs.gather(-1, target).squeeze(-1)

    mask = (x[:, 1:] != tokenizer.pad_token_id).float()

    return (token_logprob * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

def kl_penalty(logits, ref_logits):
    p = torch.log_softmax(logits, dim=-1)
    q = torch.softmax(ref_logits, dim=-1)

    return (q * (torch.log(q + 1e-8) - p)).sum(-1).mean()

def entropy_bonus(logits):
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    return -(probs * torch.log(probs + 1e-8)).sum(-1).mean()

def entropy_bonus_from_logits(logits):
    # logits: [B, T, V]
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean()

@torch.no_grad()
def evaluate_grpo(model, loader, tokenizer, device, config):

    model.eval()
    total_reward = 0
    total = 0

    for batch in loader:

        p = batch["question"].to(device)
        a = batch["answer"].to(device)

        x = sample(model, p, tokenizer)

        prompt_len = p.size(1)

        for i in range(p.size(0)):

            pred = tokenizer.decode(
                x[i, prompt_len:],
                skip_special_tokens=True
            )

            gt = tokenizer.decode(
                a[i],
                skip_special_tokens=True
            )

            total_reward += reward_fn(pred, gt)
            total += 1

    return total_reward / total

def train_one_epoch(
    model,
    ref_model,
    tokenizer,
    loader,
    optimizer,
    device,
    config,
    global_step
):

    model.train()
    ref_model.eval()
    running_reward = 0.0
    running_loss = 0.0

    for step, batch in enumerate(loader):

        p = batch["question"].to(device)
        a = batch["answer"].to(device)

        B, K = p.size(0), config.K

        prompts = p.repeat_interleave(K, dim=0)
        answers = a.repeat_interleave(K, dim=0)

        optimizer.zero_grad(set_to_none=True)
        # print("prompts: ",prompts.shape)

        # ==================================================
        # 1. Rollout generation
        # ==================================================
        x = sample(model, prompts, tokenizer)
        # print("x: ",x.shape)


        # ==================================================
        # 2. Logprob under current model
        # ==================================================
        logp = compute_logprob(model, x, tokenizer)
        # print("logp: ",logp.shape)

        # ==================================================
        # 3. Logprob under reference model (KL anchor)
        # ==================================================
        with torch.no_grad():
            ref_logits = ref_model(x[:, :-1])
        cur_logits = model(x[:, :-1])

        kl = kl_penalty(cur_logits, ref_logits)
        ent = entropy_bonus_from_logits(cur_logits)

        # ==================================================
        # 4. Rewards
        # ==================================================
        rewards = []

        prompt_len = p.size(1)

        for i in range(B * K):
            pred = tokenizer.decode(
                x[i, prompt_len:],
                skip_special_tokens=True
            )

            gt = tokenizer.decode(
                answers[i],
                skip_special_tokens=True
            )

            rewards.append(reward_fn(pred, gt))

        rewards = torch.tensor(rewards, device=device).view(B, K)

        # ==================================================
        # 5. GRPO advantage 
        # ==================================================
        baseline = rewards.mean(dim=1, keepdim=True)
        advantages = (rewards - baseline)

        advantages = advantages.reshape(B * K)

        # normalize advantages 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ==================================================
        # 6. Loss
        # ==================================================
        loss = -(logp * advantages).mean()

        loss = loss + 0.01 * kl - 0.001 * ent

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        running_reward += rewards.mean().item()
        running_loss += loss.item()

        # if step % 10 == 0:
        print(
            f"[GRPO] step {step} "
            f"loss {loss.item():.4f} "
            f"kl {kl.item():.4f} "
            f"reward {rewards.mean().item():.3f}"
        )

        global_step += 1

    return (
        running_loss / len(loader),
        running_reward / len(loader),
        global_step,
    )

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

    plt.plot(df["epoch"], df["train_reward"], label="Train Reward")
    plt.plot(df["epoch"], df["val_reward"], label="Val Reward")

    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("GPRO Training Progress")
    plt.legend()
    plt.grid(True)

    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"CSV saved: {csv_path}")
    print(f"Graph saved: {img_path}")


def train(model,tokenizer,train_loader,val_loader,optimizer, device,config):

    # =========================
    # PHASE 1: SFT
    # =========================
    print("🔥 Phase 1: Warmup SFT")

    use_amp = torch.cuda.is_available()

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    for epoch in range(config.sft_epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):

            x = batch["question_answer"].to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda"):
                    logits = model(x[:, :-1])
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        x[:, 1:].reshape(-1),
                        ignore_index=tokenizer.pad_token_id
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                logits = model(x[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    x[:, 1:].reshape(-1),
                    ignore_index=tokenizer.pad_token_id
                )

                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            if step % 20 == 0:
                print(f"SFT step {step} loss {loss.item():.4f}")

        print(f"SFT epoch : {epoch+1} loss {running_loss/len(train_loader):.4f}")

    # =========================
    # PHASE 2: GRPO
    # =========================
    print("🔥 Phase 2: GRPO")

    ref_model = copy.deepcopy(model)
    ref_model.eval()

    for p in ref_model.parameters():
        p.requires_grad = False

    global_step = 0
    best_val_reward = -1e9
    history = []

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1} | Training GRPO")

        train_loss,train_reward, global_step = train_one_epoch(
            model,
            ref_model,
            tokenizer,
            train_loader,
            optimizer,
            device,
            config,
            global_step
        )

        val_reward = evaluate_grpo(model, val_loader, tokenizer, device, config)
       
        # best model tracking
        if val_reward > best_val_reward:
            best_val_reward = val_reward

            save_checkpoint(
                "checkpoints/grpo/best.pt",
                model,
                optimizer,
                epoch+1,
                global_step
            )
        history.append({
                            "epoch": epoch + 1,
                            "train_reward": train_reward,
                            "val_reward": val_reward
                        })

        print(f"Epoch {epoch+1} | Train loss {train_loss:.4f} | Train Reward {train_reward:.4f}| Val Reward {val_reward:.4f}")
    save_history(history, "gpro_history")




