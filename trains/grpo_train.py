import re
from typing import Any
import torch
import numpy as np
from torch.nn import functional as F
from trains.checkpoint import save_checkpoint


@torch.no_grad()
def sample(model, x, tokenizer, max_new_tokens=128, temperature=0.8):
    model.eval()
    max_len = model.pos_emb.num_embeddings

    for _ in range(max_new_tokens):
        if x.size(1) >= max_len:
            break

        logits = model(x)

        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, 1)

        x = torch.cat([x, next_token], dim=1)

    # for _ in range(max_new_tokens):

    #     ctx = x[:, -model.pos_emb.num_embeddings:]

    #     logits = model(ctx)

    #     logits = logits[:, -1, :] / temperature

    #     probs = F.softmax(logits, dim=-1)

    #     next_token = torch.multinomial(probs, 1)

    #     x = torch.cat([x, next_token], dim=1)

        # if next_token.item() == tokenizer.eos_token_id:
        #     break
        if (next_token == tokenizer.eos_token_id).all():
            break

    return x
  

def extract_number(text):
    # print(text)
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None

def reward_fn(pred, gt):

    pred_num = extract_number(pred)
    gt_num = extract_number(gt)

    if pred_num is None or gt_num is None:
        return -0.5

    try:
        pred_num = float(pred_num)
        gt_num = float(gt_num)
    except Exception:
        return -0.5

    if abs(pred_num - gt_num) < 1e-6:
        return 1.0

    error = abs(pred_num - gt_num)

    return -0.01 * error

def compute_logprob(model, x,tokenizer):
    logits = model(x[:, :-1])

    log_probs = F.log_softmax(logits, dim=-1)

    target = x[:, 1:].unsqueeze(-1)

    token_logprob = log_probs.gather(-1, target).squeeze(-1)

    # ❗ mask padding
    mask = (x[:, 1:] != tokenizer.pad_token_id).float()

    token_logprob = token_logprob * mask

    return token_logprob.sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def kl_penalty(logits, ref_logits):
    p = F.log_softmax(logits, dim=-1)
    q = F.softmax(ref_logits, dim=-1)

    kl = (q * (torch.log(q + 1e-8) - p)).sum(-1)

    return kl.mean()


def entropy_bonus(logits):
    logits = logits[:, -1, :]
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(-1).mean()

@torch.no_grad()
def evaluate_grpo(
    model,
    val_loader,
    tokenizer,
    device,
    config
):

    model.eval()

    total_reward = 0.0
    total_examples = 0

    for batch in val_loader:

        p = batch["question"].to(device)
        a = batch["answer"].to(device)

        B = p.size(0)

        batch_rewards = []

        x = sample(
            model,
            p.clone(),
            tokenizer
        )

        prompt_len = p.size(1)

        for i in range(B):

            pred_text = tokenizer.decode(
                x[i, prompt_len:],
                skip_special_tokens=True
            )

            gt_text = tokenizer.decode(
                a[i],
                skip_special_tokens=True
            )

            r = reward_fn(
                pred_text,
                gt_text
            )

            batch_rewards.append(r)

        total_reward += sum(batch_rewards)
        total_examples += B

    model.train()

    return total_reward / total_examples

def train_one_epoch(
    model,
    tokenizer,
    loader,
    optimizer,
    device,
    config,
    global_step,
):

    running_reward = 0.0
    running_loss = 0.0

    use_amp = torch.cuda.is_available()

    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")

    model.train()

    for step, batch in enumerate(loader):

        p = batch["question"].to(device)      # [B, T]
        a = batch["answer"].to(device)        # [B, T]

        B = p.size(0)
        K = config.K

        optimizer.zero_grad(set_to_none=True)

        # ==================================================
        # Create K rollouts per prompt
        # ==================================================

        prompts = p.repeat_interleave(K, dim=0)      # [B*K, T]
        answers = a.repeat_interleave(K, dim=0)      # [B*K, T]

        if use_amp:

            with torch.autocast(device_type="cuda"):

                # --------------------------------------
                # Generate rollouts
                # --------------------------------------
                x = sample(
                    model,
                    prompts.clone(),
                    tokenizer
                )                                    # [B*K, T_new]

                logits = model(x)

                logp = compute_logprob(
                    model,
                    x,
                    tokenizer
                )                                    # [B*K]

                # --------------------------------------
                # Rewards
                # --------------------------------------
                reward_list = []

                for i in range(B * K):

                    pred_text = tokenizer.decode(
                        x[i],
                        skip_special_tokens=True
                    )

                    gt_text = tokenizer.decode(
                        answers[i],
                        skip_special_tokens=True
                    )

                    r = reward_fn(
                        pred_text,
                        gt_text
                    )

                    reward_list.append(r)

                rewards = torch.tensor(
                    reward_list,
                    dtype=torch.float32,
                    device=device
                )                                    # [B*K]

                # --------------------------------------
                # Reshape for GRPO
                # --------------------------------------
                rewards = rewards.view(B, K)
                logp = logp.view(B, K)

                baseline = rewards.mean(
                    dim=1,
                    keepdim=True
                )

                advantages = rewards - baseline

                # --------------------------------------
                # KL / Entropy
                # --------------------------------------
                kl = torch.tensor(
                    0.0,
                    device=device
                )

                # ent = entropy_bonus(logits)

                # loss = (
                #     -(logp * advantages).mean()
                #     + 0.01 * kl
                #     - 0.001 * ent
                # )
                loss = -(logp * advantages).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            x = sample(
                model,
                prompts.clone(),
                tokenizer
            )

            logits = model(x)

            logp = compute_logprob(
                model,
                x,
                tokenizer
            )
            print("logp: ", logp)

            reward_list = []

            for i in range(B * K):

                pred_text = tokenizer.decode(
                    x[i],
                    skip_special_tokens=True
                )

                gt_text = tokenizer.decode(
                    answers[i],
                    skip_special_tokens=True
                )

                r = reward_fn(
                    pred_text,
                    gt_text
                )

                reward_list.append(r)

            rewards = torch.tensor(
                reward_list,
                dtype=torch.float32,
                device=device
            )

            rewards = rewards.view(B, K)
            logp = logp.view(B, K)
            print("rewards: ", rewards)
            print("logp: ", logp)
            baseline = rewards.mean(
                dim=1,
                keepdim=True
            )
            print("baseline: ", baseline)
            advantages = rewards - baseline
            print("advantages: ", advantages)

            kl = torch.tensor(
                0.0,
                device=device
            )

            # ent = entropy_bonus(logits)

            # loss = (
            #     -(logp * advantages).mean()
            #     + 0.01 * kl
            #     - 0.001 * ent
            # )
            loss = -(logp * advantages).mean()

            loss.backward()
            optimizer.step()

        batch_reward = rewards.mean().item()

        running_reward += batch_reward
        running_loss += loss.item()

        if step % 5 == 0:

            print(
                f"GRPO step {step} "
                f"loss {loss.item():.4f} "
                f"reward {batch_reward:.4f}"
            )

            global_step += 1

    return (
        running_loss / len(loader),
        running_reward / len(loader),
        global_step,
    )

def train(model,tokenizer,train_loader,val_loader,optimizer, device,config):

    # =========================
    # PHASE 1: SFT
    # =========================
    print("🔥 Phase 1: Warmup SFT")

    use_amp = torch.cuda.is_available()

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")

    for step, batch in enumerate(train_loader):

        if step >= 20:
            break

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

        if step % 10 == 0:
            print(f"SFT step {step} loss {loss.item():.4f}")

    # =========================
    # PHASE 2: GRPO
    # =========================
    print("🔥 Phase 2: GRPO")
    global_step = 0
    best_val_reward = -1e9

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1} | Training GRPO")

        train_loss,train_reward, global_step = train_one_epoch(
            model,
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

        print(f"Epoch {epoch+1} | Train loss {train_loss:.4f} | Train Reward {train_reward:.4f}| Val Reward {val_reward:.4f}")




