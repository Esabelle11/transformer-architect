import torch
from trains.dpo_train import train
from data import dataset_retrieve
from model.dpo_transformer import DPO_transformer
from device import init_weights
import copy

def run_dpo_training(device,config):
    train_loader, val_loader, vocab_size , _= dataset_retrieve(config)

    policy_model = DPO_transformer(vocab_size,config)
    policy_model.apply(init_weights)

    reference_model = copy.deepcopy(policy_model)
    for p in reference_model.parameters():
        p.requires_grad = False
    reference_model.eval()
    
    policy_model = policy_model.to(device)
    reference_model = reference_model.to(device)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.lr)

    train(policy_model,reference_model, train_loader, val_loader, optimizer, device,config)