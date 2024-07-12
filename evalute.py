import torch.utils
import torch.utils.data
from dataset import separate_train_test_data, imageCaptionDataset
from transformers import AutoModelForCausalLM
from pathlib import Path
import torch.nn as nn
import torch
import pandas as pd
from tqdm import tqdm
from torchmetrics.text import WordErrorRate


def model(model_name, check_point_path):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if not check_point_path:
        return model.eval()
    weights = torch.load(check_point_path)
    if 'state_dict' in weights:
        model.load_state_dict(weights['state_dict'])
    else:
        model.load_state_dict(weights)

    return model.eval()

def test(model, test_datatset, batch_size):
    test_loader = torch.utils.data.DataLoader(test_datatset, batch_size=batch_size, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    output = {'ground_truth': [], 'generated_text': []}
    for inputs in tqdm(test_loader):
        pixel_values = inputs['pixel_values'].to(model.device)
        generate_ids = model.generate(pixel_values=pixel_values, max_length=256)
        generate_texts = test_datatset.decode(generate_ids)
        output['ground_truth'].extend(inputs['description'])
        output['generated_text'].extend(generate_texts)
    pd.DataFrame(output).to_csv('results.csv', index=False)
    wer = WordErrorRate()
    return wer(output['generated_text'], output['ground_truth'])

def main(model_name, check_point, data_csv, dataset_folder, batch_size):
    m=model(model_name, check_point)
    train_data, test_data = separate_train_test_data(data_csv)
    dataset = imageCaptionDataset(model_name, test_data, dataset_folder)
    metric = test(m, dataset, batch_size)
    print(metric)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
                    prog='image caption evaluation ',
                    description='image caption evaluation ')
    parser.add_argument('-data', '--data-csv', type=str, required=True, help='csv files images and captions')
    parser.add_argument('-m', '--model-name', default='microsoft/git-base-textcaps')
    parser.add_argument('-batch-size', '--batch-size', type=int, default= 1, help='number of images to download for training dataset')
    parser.add_argument('-dataset-folder', '--dataset-folder', type=str, default= "fashion_images", help="dataset root folder")
    parser.add_argument('-check-point', '--check-point', type=str, default= None, help="dweight file path")
    args = parser.parse_args()
    main(args.model_name,args.check_point,args.data_csv,args.dataset_folder, args.batch_size)


