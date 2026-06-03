from pandas.core.missing import F
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset ,concatenate_datasets
from torch.nn.utils.rnn import pad_sequence
from functools import partial

class GRPODataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row["question"]
        answer = row["answer"]
        question_answer = f"{question} {answer}"

        # Combine question + answer encoding
        qa_enc = self.tokenizer(
            question_answer,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        # Question only encoding
        q_enc = self.tokenizer(
            question,
            truncation=True,
            padding= False,
            max_length=self.config.q_max_length,
            return_tensors="pt",
        )

        # Answer only encoding
        a_enc = self.tokenizer(
            answer,
            truncation=True,
            padding=False,
            max_length=self.config.q_max_length,
            return_tensors="pt",
        )

        return {
            "question_answer": qa_enc["input_ids"].squeeze(0),
            # "question_answer_mask": qa_enc["attention_mask"].squeeze(0),
            "question": q_enc["input_ids"].squeeze(0),
            # "question_mask": q_enc["attention_mask"].squeeze(0),
            "answer": a_enc["input_ids"].squeeze(0),
            # "answer_mask": a_enc["attention_mask"].squeeze(0),
        }


class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        # Handle cases where data is passed as a HF dataset or raw dict slice
        self.texts = data["text"]
        self.labels = data["label"]
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class DPODataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # Use fallback if system prompt isn't present
        system_prompt = row.get("system", "")
        prompt = f"{system_prompt}\n{row['question']}".strip()
        chosen = f"{prompt}\n{row['chosen']}"
        rejected = f"{prompt}\n{row['rejected']}"

        chosen_enc = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        rejected_enc = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


def collate_fn(batch, pad_id):
    return {
        "question": pad_sequence(
            [b["question"] for b in batch],
            batch_first=True,
            padding_value=pad_id
        ),
        "answer": pad_sequence(
            [b["answer"] for b in batch],
            batch_first=True,
            padding_value=pad_id
        ),
        "question_answer": pad_sequence(
            [b["question_answer"] for b in batch],
            batch_first=True,
            padding_value=pad_id
        ),
    }

def dataset_retrieve(config):
    """
    Retrieves and prepares DataLoaders dynamically based on configuration parameters.
    """
    if config.project_name == "DPO":
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size

        # Intel/orca_dpo_pairs has only 'train' split. Slicing manually.
        dataset_train = load_dataset("Intel/orca_dpo_pairs", split="train[:1000]")
        dataset_val = load_dataset("Intel/orca_dpo_pairs", split="train[1000:1200]")
        
        train_ds = DPODataset(dataset_train, tokenizer, config)
        val_ds = DPODataset(dataset_val, tokenizer, config)
        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)  # Fixed return variable name mismatch


    elif config.project_name == "GRPO": 
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size

        # GSM8K splits are 'train' and 'test'
        dataset_train = load_dataset("openai/gsm8k", "main", split="train[:1000]")
        dataset_val = load_dataset("openai/gsm8k", "main", split="train[1000:120]")
        
        # Fixed mapping to call GRPODataset instead of IMDBDataset
        train_ds = GRPODataset(dataset_train, tokenizer, config)
        val_ds = GRPODataset(dataset_val, tokenizer, config)
        collate = partial(collate_fn, pad_id=tokenizer.pad_token_id)

        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, collate_fn=collate)  # Fixed return variable name mismatch


    elif config.project_name == "BERT":
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model_name)
        # tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        vocab_size = tokenizer.vocab_size

        # print("pad_token:", tokenizer.pad_token)
        # print("eos_token:", tokenizer.eos_token)
        # print("pad_token_id:", tokenizer.pad_token_id)

        dataset = load_dataset("mteb/imdb", split="train")

        neg = dataset.filter(lambda x: x["label"] == 0)
        pos = dataset.filter(lambda x: x["label"] == 1)

        neg = neg.shuffle(seed=42)
        pos = pos.shuffle(seed=42)

        dataset_train = concatenate_datasets([
            neg.select(range(500)),
            pos.select(range(500))
        ]).shuffle(seed=42)

        dataset_val = concatenate_datasets([
            neg.select(range(500, 600)),
            pos.select(range(500, 600))
        ]).shuffle(seed=42)

        # dataset_train = load_dataset("mteb/imdb", split="train[:200]")
        # dataset_val = load_dataset("mteb/imdb", split="train[1000:1050]")
        
        train_ds = IMDBDataset(dataset_train, tokenizer, config)
        val_ds = IMDBDataset(dataset_val, tokenizer, config)

        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)  # Fixed return variable name mismatch

    else:
        raise ValueError(f"Unsupported project_name configuration: {config.project_name}")


    return train_loader, val_loader, vocab_size,tokenizer