

## 🚀 Transformer Architect

A modular PyTorch framework for implementing, training, and experimenting with modern Transformer architectures from first principles.
<!-- 
[GitHub Repository](https://github.com/Esabelle11/transformer-architect?utm_source=chatgpt.com) -->

### Highlights

* Encoder-only Transformers (BERT-style)
* Decoder-only GPT-style models
* Preference learning (DPO)
* Reinforcement learning (GRPO)
* Mixed Precision (AMP) training
* Multi-GPU training support
* Modular training engine design
* Config-driven experiments
* Hugging Face dataset integration


This project is designed for deep understanding of LLM internals, not just high-level API usage.

## 📚 Table of Contents

- [Project Vision](##-📖-Project-Vision)
- [Architechture](##-🏗️-Architecture)
- [Implemented Models](##-🧠-Implemented-Models)
- [Quick Start](##-⚡-Quick-Start)
- [1. Encoder Transformer (BERT-style Classification Model)](##-🧠-1.-Encoder-Transformer-(BERT-style-Classification-Model))
- [2. Decoder Transformer (GPT-style Backbone for DPO & GRPO)](##-🧠-2.-Decoder-Transformer-(GPT-style-Backbone-for-DPO-&-GRPO))
- [3. DPO (Direct Preference Optimization)](##-🤖-3.-DPO-(Direct-Preference-Optimization))
- [4. GRPO (Group Relative Policy Optimization)](##-🎯-4.-GRPO-(Group-Relative-Policy-Optimization))
- [5. Evaluation System](##-📊-5.-Evaluation-System)
- [6. Shared Engineering Features](##-⚡-6.-Shared-Engineering-Features)
- [Roadmap](##-🛣️-Roadmap)


---

## 📖 Project Vision

This repository follows a progressive learning structure:
```text
Transformer Basics
      ↓
Encoder Models (BERT-style classification)
      ↓
Decoder Models (GPT-style generation)
      ↓
Preference Learning (DPO)
      ↓
Reinforcement Learning (GRPO)
```
👉 Goal: Understand how modern LLMs evolve from supervised learning → alignment → RL optimization.

---

## 🏗️ Architecture

```text
transformer-architect/
│
├── configs/          # YAML experiment configs
├── models/
│   ├── bert_transformer.py
│   ├── grpo_transformer.py
│   └── dpo_transformer.py
│
├── engines/         # Load and initialize model and training progress
│   ├── bert_engine.py
│   ├── dpo_engine.py
│   └── grpo_engine.py
│
├── trains/
│   ├── bert_train.py
│   ├── dpo_train.py
│   └── grpo_train.py
│   └── checkpoint.py
│
├── device.py      # Device and initialize function
├── data.py        # Dataset loaders
├── main.py        # The main file 
└── requirements.txt
```

---

## 🧠 Implemented Models

### BERT

Features:

* Encoder-only Transformer
* Masked Language Modeling
* Sequence Classification
* IMDB sentiment training example

### DPO

Features:

* Preference learning
* Policy vs Reference model
* Pairwise ranking loss

### GRPO

Features:

* Group-based reward optimization
* Multiple response sampling
* Relative reward normalization

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/Esabelle11/transformer-architect.git

cd transformer-architect

pip install -r requirements.txt
```

### Train BERT

```bash
python main.py --config configs/bert_classification.yaml
```

### Train DPO

```bash
python main.py --config configs/dpo_alignment.yaml
```

### Train GRPO

```bash
python main.py --config configs/grpo_reasoning.yaml
```
---

## 🧠 1. Encoder Transformer (BERT-style Classification Model)

### 📌 Architecture
This model is an **encoder-only Transformer for sequence classification**.

It is NOT full BERT (no MLM / NSP).

#### Model Flow

```
Input Tokens
     ↓
Token Embedding + Position Embedding
     ↓
Encoder Block × N
     ↓
Mean Pooling
     ↓
Linear Classifier
     ↓
Logits
```

### 🔧 Encoder Block Structure

Each block contains:

* Multi-Head Self Attention
* Residual Connections
* LayerNorm
* FeedForward Network


### ⚙️ Training Objective

```
CrossEntropyLoss(logits, labels)
```


### ⚡ Training Loop

```
Forward Pass
   ↓
Compute Loss
   ↓
Backward Pass
   ↓
Optimizer Step (AdamW)
   ↓
Validation Accuracy Tracking
   ↓
Best Model Checkpoint Saving
```

### 🧪 Key Implementation Details

* Mean pooling instead of CLS token
* Attention mask support
* Mixed precision training (AMP optional)
* Best checkpoint saving based on validation accuracy

### 🎯 Weight Initialization

All linear + embedding layers use:

```python
torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
```

This follows GPT/BERT-style initialization for stable training.

---

## 🧠 2. Decoder Transformer (GPT-style Backbone for DPO & GRPO)

This model is used in both:

* DPO
* GRPO

### 📌 Architecture

```
Input Tokens
     ↓
Token Embedding + Position Embedding
     ↓
Decoder Block × N
     ↓
LayerNorm
     ↓
LM Head
     ↓
Logits (vocab distribution)
```


### 🔧 Decoder Block Structure

Each block contains:

* Causal Self-Attention (masked)
* FeedForward Network
* Residual Connections
* LayerNorm

#### Important Design Choice

```text
Causal Mask (lower triangular)
→ prevents future token leakage
```


### ⚙️ Training Objective (Used in DPO / GRPO)

* Autoregressive next-token prediction
* Log-probability extraction for sequences

---

## 🤖 3. DPO (Direct Preference Optimization)

### 📌 Concept

DPO trains a model using:

```
(Chosen response, Rejected response)
```

WITHOUT reinforcement learning or reward models.


### 🧠 DPO Architecture Flow

```
Prompt
  ↓
Policy Model (Trainable)
  ↓
Chosen Log Prob   Rejected Log Prob
  ↓
Reference Model (Frozen)
  ↓
Chosen Ref Log Prob   Rejected Ref Log Prob
```

### 🔢 Key Computation

#### Sequence Log Probability

```
log π(y|x) = sum(log softmax over tokens)
```
#### DPO Loss

```
Δπ = logπθ(chosen) - logπθ(rejected)

Δref = logπref(chosen) - logπref(rejected)

Loss = -log σ(β(Δπ - Δref))
```


### ⚙️ Training Flow

```
Policy Model + Reference Model
        ↓
Compute log probabilities
        ↓
DPO loss
        ↓
Backpropagation
        ↓
AdamW update (policy only)
```

### 🧪 Key Features

* Frozen reference model
* Gradient only updates policy model
* AMP training support
* Gradient accumulation
* Best checkpoint saving based on validation loss

---

## 🎯 4. GRPO (Group Relative Policy Optimization)

### 📌 Concept

GRPO trains a model using:

* Multiple sampled outputs per prompt (K rollouts)
* Reward function
* Group-relative advantage (baseline normalization)

---

### 🧠 Architecture Flow

```
Question Prompt
      ↓
Repeat K times
      ↓
Policy Model (Sampling)
      ↓
K Generated Responses
      ↓
Reward Function
      ↓
Group Baseline (Mean Reward)
      ↓
Advantage = Reward - Mean
```


### ⚙️ Reward Function

Your implementation:

```
Extract number from response
Compare with ground truth
```

Reward:

* +1 → correct answer
* negative penalty → incorrect or numeric error



### 📊 Group Advantage

```
A = r - mean(r)
```

This stabilizes learning by removing reward scale bias.

### 🔢 GRPO Objective

```
Loss = -(logπ(y) × Advantage).mean()
```

This is essentially:

> REINFORCE with group-normalized baseline

### ⚙️ Training Pipeline

```
SFT Warmup
    ↓
Generate K Rollouts
    ↓
Compute Rewards
    ↓
Compute Baseline
    ↓
Compute Advantages
    ↓
Policy Gradient Update
```

### 🧪 Key Features

* Two-stage training (SFT → GRPO)
* K-rollout sampling per prompt
* No value network
* No PPO clipping
* Optional KL + entropy (disabled in current version)
* AMP support
* Reward-driven learning loop

---

## 📊 5. Evaluation System

### BERT Evaluation

* Accuracy-based (classification)

```
Accuracy = correct / total
```


### GRPO Evaluation

```
Sample response
   ↓
Reward function
   ↓
Average reward over dataset
```

---

<!-- ## 📊 Current Results

| Model | Dataset            | Metric              |
| ----- | ------------------ | ------------------- |
| BERT  | IMDB               | Accuracy            |
| DPO   | Preference Dataset | Reward              |
| GRPO  | GSM8K              | Validation Accuracy | -->

## ⚡ 6. Shared Engineering Features

### 🧠 Mixed Precision Training

* torch.autocast
* GradScaler


### 💾 Checkpointing

* Best model saving based on:

  * Accuracy (BERT)
  * Loss (DPO)
  * Reward (GRPO)

### ⚙️ Optimizer

* AdamW across all models


### 🔬 Learning Objectives

This project focuses on understanding:

* Attention mechanisms
* Transformer architecture
* Language model training
* Preference optimization
* Reinforcement learning for LLMs
* Distributed training
* Mixed precision training

---

## 🛣️ Roadmap

* [x] Encoder Transformer
* [x] GPT Decoder
* [x] DPO
* [x] GRPO
* [ ] PPO (future)
* [ ] LoRA fine-tuning
* [ ] Multi-GPU training
* [ ] MoE architecture

---

## ⭐ Why This Repository?

Most repositories either:

* use Hugging Face without explaining internals, or
* implement only the original Transformer.

This repository bridges the gap by showing how modern Transformer systems evolve from foundational architectures to alignment methods such as DPO and GRPO in a single codebase.

---



