import torch
# from transformers.models import configuration_albert
from trains.grpo_train import train
from data import dataset_retrieve
from model.grpo_transformer import grpo_transformer
from device import init_weights

def run_grpo_training(device,config):
    train_loader, val_loader, vocab_size , tokenizer= dataset_retrieve(config)

    model = grpo_transformer(vocab_size,config)
    model.apply(init_weights)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train(model, tokenizer, train_loader, val_loader, optimizer, device,config)