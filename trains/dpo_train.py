
import torch
import numpy as np
from torch.nn import functional as F
from trains.checkpoint import save_checkpoint
import os
import pandas as pd
import matplotlib.pyplot as plt


def sequence_logprob(logits, labels):

    shift_logits = logits[:, :-1, :].contiguous()

    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(
        shift_logits,
        dim=-1
    )

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs.sum(dim=-1)



def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1
):

    policy_ratio = (
        policy_chosen_logps
        - policy_rejected_logps
    )

    ref_ratio = (
        ref_chosen_logps
        - ref_rejected_logps
    )

    loss = -F.logsigmoid(
        beta * (policy_ratio - ref_ratio)
    )

    return loss.mean()


def train_one_epoch(policy_model,reference_model,loader,optimizer,device,config,global_step):

    policy_model.train()
    reference_model.eval()

    total_loss = 0.0

    use_amp = torch.cuda.is_available()

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):

        chosen_ids = batch["chosen_input_ids"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)

        # =========================
        # FORWARD
        # =========================

        if use_amp:

            with torch.autocast(device_type="cuda"):

                # POLICY
                chosen_logits = policy_model(chosen_ids)
                rejected_logits = policy_model(rejected_ids)

                policy_chosen_logps = sequence_logprob(
                    chosen_logits,
                    chosen_ids
                )

                policy_rejected_logps = sequence_logprob(
                    rejected_logits,
                    rejected_ids
                )

                # REFERENCE (frozen)
                with torch.no_grad():

                    ref_chosen_logits = reference_model(chosen_ids)
                    ref_rejected_logits = reference_model(rejected_ids)

                    ref_chosen_logps = sequence_logprob(
                        ref_chosen_logits,
                        chosen_ids
                    )

                    ref_rejected_logps = sequence_logprob(
                        ref_rejected_logits,
                        rejected_ids
                    )

                loss = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=config.beta
                )

                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

        else:

            # POLICY
            chosen_logits = policy_model(chosen_ids)
            rejected_logits = policy_model(rejected_ids)

            policy_chosen_logps = sequence_logprob(
                chosen_logits,
                chosen_ids
            )

            policy_rejected_logps = sequence_logprob(
                rejected_logits,
                rejected_ids
            )

            # REFERENCE (frozen)
            with torch.no_grad():

                ref_chosen_logits = reference_model(chosen_ids)
                ref_rejected_logits = reference_model(rejected_ids)

                ref_chosen_logps = sequence_logprob(
                    ref_chosen_logits,
                    chosen_ids
                )

                ref_rejected_logps = sequence_logprob(
                    ref_rejected_logits,
                    rejected_ids
                )

            loss = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=config.beta
            )

            loss = loss / config.accumulation_steps

            loss.backward()

        total_loss += loss.item()

        # =========================
        # OPTIMIZER STEP
        # =========================

        should_step = (
            (step + 1) % config.accumulation_steps == 0
            or
            (step + 1) == len(loader)
        )

        if should_step:

            if use_amp:

                scaler.step(optimizer)
                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            print(
                f"Step {global_step} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader)

    return avg_loss, global_step

def evaluate(
    policy_model,
    reference_model,
    val_loader,
    device,
    config,
):

    policy_model.eval()
    reference_model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:

            chosen_ids = batch["chosen_input_ids"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)

            chosen_logits = policy_model(chosen_ids)
            rejected_logits = policy_model(rejected_ids)

            policy_chosen_logps = sequence_logprob(chosen_logits, chosen_ids)
            policy_rejected_logps = sequence_logprob(rejected_logits, rejected_ids)

            ref_chosen_logits = reference_model(chosen_ids)
            ref_rejected_logits = reference_model(rejected_ids)

            ref_chosen_logps = sequence_logprob(ref_chosen_logits, chosen_ids)
            ref_rejected_logps = sequence_logprob(ref_rejected_logits, rejected_ids)

            loss = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=config.beta
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)


    print(f"[Validation] Loss: {avg_loss:.4f}")

    return avg_loss

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

def train(
        policy_model,
        reference_model,
        train_loader,
        val_loader,
        optimizer,
        device,
        config
    ):

    global_step = 0
    best_val_loss = float("inf")
    history = []

    for epoch in range(config.epochs):

        train_loss, global_step = train_one_epoch(
            policy_model,
            reference_model,
            train_loader,
            optimizer,
            device,
            config,
            global_step
        )

        val_loss = evaluate(
            policy_model,
            reference_model,
            val_loader,
            device,
            config
        )

       
        # best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                "checkpoints/dpo/best.pt",
                policy_model,
                optimizer,
                epoch,
                global_step
            )

        history.append({
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "val_loss": val_loss
                        })

        print(f"Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    save_history(history, "dpo_history")

