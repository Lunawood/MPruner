import os
import time
import copy
import argparse
import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AdamW
from tqdm import tqdm



class QGDataset(Dataset):

    def __init__(self, tokenizer, file_path, max_len_input=512, max_len_output=128):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(file_path)
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output
        self.context_column = 'context'
        self.answer_column = 'answer'
        self.question_column = 'question'
        self.inputs = []
        self.targets = []
        self._load_data()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        target_ids = self.targets[index]['input_ids'].squeeze()
        source_mask = self.inputs[index]['attention_mask'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()
        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100
        return {'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids, 'target_mask': target_mask, 'labels': labels}

    def _load_data(self):
        for idx in tqdm(range(len(self.data))):

            context, answer, target = self.data.loc[idx, self.context_column], self.data.loc[idx, self.answer_column], self.data.loc[idx, self.question_column]
            input_text = '<answer> %s <context> %s ' % (answer, context)
            target = str(target)

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_text],
                max_length=self.max_len_input,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target],
                max_length=self.max_len_output,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)




class T5FineTuner(pl.LightningModule):

    def __init__(self, model, tokenizer, train_set,test_set,args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_set = train_set
        self.test_set = test_set

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels, 
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def train_dataloader(self):

        return DataLoader(self.train_set, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.args.learning_rate, eps=self.args.eps)

