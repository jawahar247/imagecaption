from transformers import AutoProcessor
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os


def train_and_test_dataset(csv_file, model_name, data_folder=None):
    df = pd.read_csv(csv_file)
    df = df.sample(frac=1).reset_index(drop=True)
    train_data = df[0:int(len(df)*0.9)]
    test_data = df[int(len(df)*0.9):]
    train_data = imageCaptionDataset(model_name, train_data, data_folder)
    test_data = imageCaptionDataset(model_name, test_data, data_folder)
    return train_data, test_data

class imageCaptionDataset(Dataset):
    def __init__(self, model_name, data, data_folder=None) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.data_folder = data_folder
        self.dataset = data

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset.iloc[index]
        try:
            if self.data_folder:
                path = os.path.join(self.data_folder, item['image'])
            else:
                path = item['image']
            
            image = Image.open(path)
            text = item['description']
        except Exception as e:
            print(f"{e}")
            image = Image.open('dataset/122349_Gold_0.jpeg')
            text = "subtly futuristic and edgy this liquid metal cuff bracelet is shaped from sculptural rectangular link"
            path = "dataset/122349_Gold_0.jpeg"
        
        inputs = self.processor(images=image, text=text, padding="max_length", return_tensors='pt')
        inputs = {k:v.squeeze() for k,v in inputs.items()}
        inputs.update({'labels': inputs['input_ids'], 'description':text, 'image_path': path})
        return inputs
    
    def decode(self, generate_ids):
        return self.processor.batch_decode(generate_ids, skip_special_tokens=True)