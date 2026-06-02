import torch
from trains.bert_train import train
from data import dataset_retrieve
from model.bert_transformer import EncoderTransformer
from device import init_weights

def run_bert_training(device,config):
    train_loader, val_loader, vocab_size , tokenizer= dataset_retrieve(config)

    model = EncoderTransformer(vocab_size,config)
    model.apply(init_weights)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train(model, train_loader, val_loader, optimizer, device,config)