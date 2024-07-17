from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from typing import  Tuple
from image_caption_dataset import train_and_test_dataset
from transformers import AutoModelForCausalLM
from torcheval.metrics import WordErrorRate
import pandas as pd
from tqdm import tqdm

def prepare_const() -> dict:
    """Data and model directory + Training hyperparameters"""
    data_root = Path("data")
    trained_models = Path("trained_models")

    if not data_root.exists():
        data_root.mkdir()

    if not trained_models.exists():
        trained_models.mkdir()

    const = dict(
        data_root=data_root,
        trained_models=trained_models,
        total_epochs=2,
        batch_size=2,
        lr=0.0001,  # learning rate
        momentum=0.9,
        lr_step_size=2,
        save_every=1,
    )

    return const




class TrainerSingle:
    def __init__(
        self,
        gpu_id: int,
        model: nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
    ):
        self.gpu_id = gpu_id
        self.const = prepare_const()
        self.model = model.to(self.gpu_id)
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.const["lr"],
            momentum=self.const["momentum"],
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, self.const["lr_step_size"]
        )

        self.valid_acc = WordErrorRate().to(self.gpu_id)

    def _run_batch(self, inputs) -> float:
        self.optimizer.zero_grad()
        out = self.model(
            input_ids = inputs['input_ids'].to(self.model.device),
            attention_mask = inputs['attention_mask'].to(self.model.device),
            pixel_values = inputs['pixel_values'].to(self.model.device),
            labels = inputs['labels'].to(self.model.device),
        )
        loss = out.loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch: int):
        loss = 0.0
        for inputs in tqdm(self.trainloader):
            loss_batch = self._run_batch(inputs)
            loss += loss_batch
        self.lr_scheduler.step()

        print(
            f"{'-' * 90}\n[GPU{self.gpu_id}] Epoch {epoch:2d} | Batchsize: {self.const['batch_size']} | Steps: {len(self.trainloader)} | LR: {self.optimizer.param_groups[0]['lr']:.4f} | Loss: {loss / len(self.trainloader):.4f}",
            flush=True,
        )


    def _save_checkpoint(self, epoch: int):
        ckp = self.model.state_dict()
        model_path = self.const["trained_models"] / f"image_caption_epoch{epoch}.pt"
        torch.save(ckp, model_path)

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.const["save_every"] == 0:
                self._save_checkpoint(epoch)
        # save last epoch
        self._save_checkpoint(max_epochs - 1)

    def test(self, test_dataset, final_model_path: str=None, ):
        if final_model_path:
            self.model.load_state_dict(torch.load(final_model_path))
        
        self.model.eval()
        output_dict = {'image_path':[] ,'generated_text': [], 'ground_truth': []}
        with torch.no_grad():
            for inputs in tqdm(self.testloader):
                out = self.model.generate(
                    pixel_values = inputs['pixel_values'].to(self.model.device), max_length=256,
                )
                generate_texts = test_dataset.decode(out)
                for i,j,k in zip(inputs['image_path'], generate_texts, inputs['description']):
                    output_dict["image_path"].append(i)
                    output_dict['generated_text'].append(j)
                    output_dict["ground_truth"].append(k)
                self.valid_acc.update(generate_texts, inputs['description'])
            pd.DataFrame(output_dict).to_csv("results.csv", index=False)
        print(
            f"[GPU{self.gpu_id}] Test Acc: {100 * self.valid_acc.compute().item():.4f}%"
        )



def image_caption_dataloders(
    trainset: Dataset, testset: Dataset, bs: int
) -> Tuple[DataLoader, DataLoader]:
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=32)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=32)

    return trainloader, testloader



def main_single(model_name: str, dataset: str, data_folder:str, gpu_id: int, final_model_path: str=None):
    const = prepare_const()
    train_dataset, test_dataset = train_and_test_dataset(dataset, model_name, data_folder)
    train_dataloader, test_dataloader = image_caption_dataloders(
        train_dataset, test_dataset, const["batch_size"]
    )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    trainer = TrainerSingle(
        gpu_id=gpu_id,
        model=model,
        trainloader=train_dataloader,
        testloader=test_dataloader,
    )
    # trainer.train(const["total_epochs"])
    trainer.test(test_dataset, "/home/jawahar/Workspace/ijawahar/imagecaption/trained_models/image_caption_epoch1.pt")



def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
                    prog='fashoin image caption training',
                    description='ifashion image caption training')
    parser.add_argument('-m', '--model-name', type=str, required=False, default= "microsoft/git-base-textcaps",help='hugging face model name to train ')
    parser.add_argument('-dataset-folder', '--dataset-folder', type=str,default= None, help="dataset root folder")
    parser.add_argument('-csv-file', '--csv-file', type=str, required=True, help="dweight file path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    gpu_id = "cuda"
    main_single(args.model_name, args.csv_file, args.dataset_folder, gpu_id)
