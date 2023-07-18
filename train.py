import os
import json
import time
import clip
import pandas
import numpy as np
import torch
import torchvision
import argparse
import torchvision.datasets as datasets
from balancedsampler import BalancedBatchSampler
from tqdm import tqdm
from utils import init_logger_writer, inform_logger_writer, close_logger_writer

model_ver = "ViT-B/16"
root_path= '.'

parser = argparse.ArgumentParser(description='train process')
parser.add_argument('--kshots', default='1', type = int)
parser.add_argument('--batch_size', default = 96, type = int)
parser.add_argument('--epochs', default = 15, type = int)
args = parser.parse_args()
KSHOTS, BATCH_SIZE, EPOCHS = args.kshots, args.batch_size, args.epochs

data_path = os.path.join(root_path, 'ImageNet-1K-16shot')
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'val')

with open(os.path.join(data_path, 'label2class.json'), 'r') as f:
    json_data = json.load(f)['label2class']
train_labels = [json_data[str(k)] for k in json_data]

model, preprocess = clip.load(model_ver)

class kshotFolder(datasets.ImageFolder):
    def __init__(self, path, transform, target_transform, k):
        super().__init__(path, transform = transform, target_transform = target_transform)
        self.samples = [v for i, v in enumerate(self.samples) if i % 16 < k]
        self.targets = [s[1] for s in self.samples]

train_dataset = kshotFolder(train_path, transform = preprocess, target_transform = lambda x:json_data[str(x)], k = KSHOTS)
train_sampler = BalancedBatchSampler(train_dataset.targets, BATCH_SIZE, 1)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler = train_sampler)

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * EPOCHS)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float()
        if p.grad != None:
            p.grad.data = p.grad.data.float()

template = 'a photo of a {}.'
best = 1e10
logger, writer = init_logger_writer(args)
time_list = [0] * 2
start_time = time.time()
for epoch in range(EPOCHS):
    time_list[0] = time.time()
    tr_loss = step = 0
    model.train()
    for i, layer in enumerate(model.children()):
        if i in [1, 2, 3]:
            for param in layer.parameters():
                param.requires_grad = False            
    for images, texts in tqdm(train_loader):
        optimizer.zero_grad()
        text = clip.tokenize([template.format(x) for x in texts]).cuda()
        images = images.cuda()
        logits_per_image, logits_per_text = model(images, text)
        ground_truth = torch.arange(len(texts)).cuda()

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        tr_loss += total_loss.item() * len(texts)
        step += len(texts)
        convert_models_to_fp32(model)
        optimizer.step()
        scheduler.step()
        clip.model.convert_weights(model)
    time_list[1] = time.time()
    tr_loss /= step
    inform_logger_writer(logger, writer, epoch + 1, tr_loss, time_list)
    if tr_loss < best:
        best = tr_loss
        torch.save(model.state_dict(), "best_model_{}.pt".format(KSHOTS))
close_logger_writer(logger, writer, start_time)
