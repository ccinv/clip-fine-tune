import argparse
import os
import json
import numpy as np

model_ver = "ViT-B/16"

root_path= '.'
data_path = os.path.join(root_path, 'ImageNet-1K-16shot')
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'val')

import torch
import torchvision
import torchvision.datasets as datasets
import clip
from tqdm import tqdm

parser = argparse.ArgumentParser(description='val process')
parser.add_argument('--kshots', default='1', type = int)
args = parser.parse_args()
KSHOTS = args.kshots

model, preprocess = clip.load(model_ver)
if k != 0:
    model.load_state_dict(torch.load("best_model_{}.pt".format(KSHOTS)))
model.cuda().eval()
test_dataset = datasets.ImageFolder(test_path, transform=preprocess)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
num_classes = len(test_dataset.classes)

with open(os.path.join(data_path, 'label2class.json'), 'r') as f:
    json_data = json.load(f)
label2class = json_data['label2class']
classnames = [v for k, v in label2class.items()]

def clip_classifier(classnames, clip_model, template):
    clip_model.eval()
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            texts = template.format(classname)
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embedding = clip_model.float().encode_text(texts)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embedding)
        clip_weights = torch.cat(clip_weights, dim=0).cuda()
    return clip_weights
template = 'a photo of a {}.'
text_classifier = clip_classifier(classnames, model, template)

top1_accuracy = 0.0
top5_accuracy = 0.0
total_samples = 0
with torch.no_grad():
    step = 0
    for images, labels in tqdm(test_loader):
        step += 1
        images = images.cuda()
        labels = labels.cuda()
        image_features = model.float().encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_classifier.t()
        probs = logits.softmax(dim=-1)
        _, predicted_labels = probs.topk(5, dim=-1)
        predicted_labels = predicted_labels.cuda()
        total_samples += labels.size(0)
        top1_accuracy += (predicted_labels[:, 0] == labels).sum().item()
        top5_accuracy += (predicted_labels == labels.view(-1, 1)).sum().item()
        if step % 100 == 0:
            print("Top-1 Accuracy: {:.2f}%".format(top1_accuracy / total_samples * 100))
            print("Top-5 Accuracy: {:.2f}%".format(top5_accuracy / total_samples * 100))
            
top1_accuracy /= total_samples
top5_accuracy /= total_samples
print("Top-1 Accuracy: {:.2f}%".format(top1_accuracy * 100))
print("Top-5 Accuracy: {:.2f}%".format(top5_accuracy * 100))

