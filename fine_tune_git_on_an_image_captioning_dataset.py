# -*- coding: utf-8 -*-


# !pip install -q git+https://github.com/huggingface/transformers.git

# !pip install -q datasets

import torch

import json,os
import PIL.Image
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
# data = pd.read_csv("data_dict.csv")
# data = data[:20]
from datasets import load_dataset
dataset = load_dataset("csv", data_files="/home/jawahar/Workspace/train_data_1000.csv",split="train")


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = self.processor(images=Image.open(item["img_path"]), text=item["description"], padding="max_length", return_tensors="pt")

        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}

        return encoding

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/git-base-textcaps")

train_dataset = ImageCaptioningDataset(dataset, processor)


item = train_dataset[0]
for k,v in item.items():
  print(k,v.shape)


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)


from PIL import Image
import numpy as np

MEAN = np.array([123.675, 116.280, 103.530]) / 255
STD = np.array([58.395, 57.120, 57.375]) / 255

unnormalized_image = (batch["pixel_values"][0].numpy() * np.array(STD)[:, None, None]) + np.array(MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
Image.fromarray(unnormalized_image)



from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textcaps")

# check_point = torch.load('/home/jawahar/Workspace/check_points/epoch_15_weights.pth.tar')
# model.load_state_dict(check_point['state_dict'])
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# image_folder = "/home/jawahar/Workspace/"
# images = pd.read_csv('/home/jawahar/Workspace/test_data_500.csv')

# for image_path, ground_truth in zip(images['img_path'], images['description']):
#   path = os.path.join(image_folder, image_path)
#   image=Image.open(path)
#   # prepare image for the model
#   inputs = processor(images=image, return_tensors="pt").to(device)
#   pixel_values = inputs.pixel_values

#   generated_ids = model.generate(pixel_values=pixel_values, max_length=256)
#   generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#   with open("with_ground_truths_after_fine_tune_image_captions.txt", 'a') as f:
#     print(f"{ground_truth}  --> {generated_caption}", file=f)

while True:
  image_path = input("please provide image path: ")
  if image_path == 'break':
     break

  image=Image.open(image_path)
  # prepare image for the model
  inputs = processor(images=image, return_tensors="pt").to(device)
  pixel_values = inputs.pixel_values

  generated_ids = model.generate(pixel_values=pixel_values, max_length=256)
  generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  print('output_text =', generated_caption)

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


# model.to(device)

# model.train()

# for epoch in range(20):
#   print("Epoch:", epoch)
#   for idx, batch in enumerate(train_dataloader):
#     input_ids = batch.pop("input_ids").to(device)
#     pixel_values = batch.pop("pixel_values").to(device)

#     outputs = model(input_ids=input_ids,
#                     pixel_values=pixel_values,
#                     labels=input_ids)

#     loss = outputs.loss
              
#     if idx%1000 == 0:
#       print("Loss:", loss.item())

#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
    
#   if epoch >=5 and epoch % 4:
#     optimizer.defaults['lr'] = optimizer.defaults['lr']/10
  
#   if epoch >=5:
#     save_state = {
#                 'epoch': epoch,
#                 'state_dict': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'version': 2,  # version < 2 increments epoch before save
#             }
#     torch.save(save_state, f"/home/jawahar/Workspace/check_points/epoch_{epoch}_weights.pth.tar")




# image_folder = "/home/jawahar/Workspace/"
# images = pd.read_csv('/home/jawahar/Workspace/test_data_500.csv')

# for image_path in images['img_path']:
#   path = os.path.join(image_folder, image_path)
#   image=Image.open(path)
#   # prepare image for the model
#   inputs = processor(images=image, return_tensors="pt").to(device)
#   pixel_values = inputs.pixel_values

#   generated_ids = model.generate(pixel_values=pixel_values, max_length=256)
#   generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  
#   with open("after_fine_tune_image_captions.txt", 'a') as f:
#     print(generated_caption, file=f)
