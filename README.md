# 🚀 transformer-architect: Unified Pipeline for BERT, DPO, and GRPO Workflows

## 📌 Project Overview
Having spent significant time building application-layer systems with LLM Agents and API integrations, I reached a point where I wanted to fundamentally understand the backend mechanics, algorithmic logic, and structural scaling of the models themselves.

While studying the original "Attention Is All You Need" Encoder-Decoder architecture, I realized its classic design is heavily bound to seq-to-seq translation tasks. To truly capture how modern NLP has evolved across different production environments, I pushed myself out of the tutorial comfort zone. Over an intensive week of engineering, I built transformer-architect: a single, unified codebase that transitions past standard implementation to explore how distinct transformer variants solve entirely different specialized tasks.

Instead of maintaining fragmented, messy experimental scripts, this repository acts as a single-point orchestrator. By modifying a declarative YAML configuration, the system dynamically routes execution to one of three core machine learning paradigms:

1. **Classic Supervised Fine-Tuning (SFT):** Semantic embedding extraction and text classification using an Encoder-only architecture (**BERT**).
2. **Offline Preference Alignment:** Reference-guided policy optimization utilizing modern Decoders(**DPO** via Llama/Qwen).
3. **Reinforcement Learning via Group Rewards:** Advanced reasoning-loop optimization without an explicit critic network(**GRPO** via Qwen/DeepSeek-style workflows).

---

## 🛠️ Key Engineering Features

* **Zero-Code Swapping:** Seamlessly toggle between embedding-based classification, pairwise preference alignment, and reinforcement learning. The core orchestrator dynamically switches engines purely based on the input runtime argument:
```
python src/main.py --config configs/your_config_filename.yaml
```
* **Production-Grade Data Factory (`data.py`):** Implements highly robust, customized PyTorch Dataset wrappers designed to dynamically tokenize, slice, pad, and structure completely different raw data shapes (e.g., contrastive pair extraction for DPO, or prompt-block collation for GRPO).
* **Agnostic Tokenization:** Leverages Hugging Face's `AutoTokenizer` primitives to handle diverse model vocabularies out of the box, ensuring seamless token processing without crashing downstream tensor collation pipelines.
* **Strict MLOps Reproducibility:** Every critical parameter—including learning rate schedules, vocabulary targets, token cutoffs, optimization bounds, and architectural variables—is entirely decoupled from the code and declared in human-readable YAML configurations.