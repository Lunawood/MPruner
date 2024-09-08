import time

import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re
from datetime import datetime
import pytorch_lightning as pl
from train_util import T5FineTuner
from train_util import QGDataset
import argparse
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from datasets import load_metric
from generate import QuestionGeneration
from sentence_embeddings import SentenceEmbeddings
from nltk.tokenize import word_tokenize
import nltk


train_file_path = "datasets/squad_train.csv"
validation_file_path = "datasets/squad_validation.csv"


class T5_MPruner:
    def __init__(
        self,
        model_name="anonymous78784949/t5-squad-QG",
        tokenizer_path=None,
        mpruned_model_path="mpruned_model.pth",
        dataset_name="squad",
        args=None,
        useHGModel=False,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.activation_enc = {}
        self.activation_dec = {}

        self.use_hg = useHGModel

        if useHGModel:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.model = torch.load(model_name)

        self.model = self.model.to(self.device)
        self.tokenizer_path = tokenizer_path

        if tokenizer_path is None:
            tokenizer_path = model_name

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

        self.dataset = load_dataset(dataset_name)  # squad
        self.test_dataset = self.dataset["validation"]
        self.model_name_list_enc = []
        self.model_name_list_dec = []
        self.cka_analysis_layer_name_list = ["T5Block"]
        self.enc_module_idx_list = []
        self.dec_module_idx_list = []
        self.enc_module_layer_list = []
        self.dec_module_layer_list = []
        self.args = args

        self.enc_activation_size = 0
        self.dec_activation_size = 0
        self.mpruned_model_path = mpruned_model_path

        for full_name, module in self.model.encoder.named_modules():
            if module.__class__.__name__ in self.cka_analysis_layer_name_list:

                class_name = ""

                if module.__class__.__name__ == "T5LayerSelfAttention":
                    class_name = "EncoderSelfAttention"
                elif module.__class__.__name__ == "T5LayerFF":
                    class_name = "EncoderFeedForward"
                elif module.__class__.__name__ == "T5Block":
                    class_name = "EncoderBlock"
                else:
                    continue

                re_name = re.sub(r"\.(\d+)", r"[\1]", full_name)

                block_idx = re.search(r"block\[(\d+)\]", re_name).group(1)

                full_var_name = "self.model.encoder." + re_name
                self.model_name_list_enc.append(full_var_name)
                regist_hook_cmd = (
                    full_var_name
                    + ".register_forward_hook(self.get_activation_enc(f'"
                    + class_name
                    + "{"
                    + block_idx
                    + "}'))"
                )

                self.enc_activation_size += 1

                eval(regist_hook_cmd)

        for full_name, module in self.model.decoder.named_modules():
            if module.__class__.__name__ in self.cka_analysis_layer_name_list:

                class_name = ""

                if module.__class__.__name__ == "T5LayerSelfAttention":
                    class_name = "DecoderSelfAttention"
                elif module.__class__.__name__ == "T5LayerFF":
                    class_name = "DecoderFeedForward"
                else:
                    class_name = "DecoderCrossAttention"

                re_name = re.sub(r"\.(\d+)", r"[\1]", full_name)

                block_idx = re.search(r"block\[(\d+)\]", re_name).group(1)

                full_var_name = "self.model.decoder." + re_name
                self.model_name_list_dec.append(full_var_name)
                regist_hook_cmd = (
                    full_var_name
                    + ".register_forward_hook(self.get_activation_dec(f'"
                    + class_name
                    + "{"
                    + block_idx
                    + "}'))"
                )
                self.dec_activation_size += 1
                eval(regist_hook_cmd)

        self.enc_cka_matrix = torch.zeros(
            self.enc_activation_size, self.enc_activation_size
        ).to(self.device)
        self.dec_cka_matrix = torch.zeros(
            self.dec_activation_size, self.dec_activation_size
        ).to(self.device)

    def get_activation_enc(self, name):
        def hook(model, input, output):
            self.activation_enc[name] = output[0].detach()

        return hook

    def get_activation_dec(self, name):
        def hook(model, input, output):
            self.activation_dec[name] = output[0].detach()

        return hook

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones((n, n), device=K.device).to(torch.float64)
        I = torch.eye(n, device=K.device)
        H = I - unit / n
        return torch.mm(H, torch.mm(K, H))

    def HSIC(self, Kx_centered, Ky_centered):
        return torch.trace(torch.mm(Kx_centered, Ky_centered))

    def linear_CKA(self, X, Y):
        Kx = torch.mm(X, X.T).to(torch.float64)
        Ky = torch.mm(Y, Y.T).to(torch.float64)

        Kx_centered = self.centering(Kx)
        Ky_centered = self.centering(Ky)
        hsic = self.HSIC(Kx_centered, Ky_centered)
        var_x = self.HSIC(Kx_centered, Kx_centered)
        var_y = self.HSIC(Ky_centered, Ky_centered)

        cka_score = hsic / (torch.sqrt(var_x) * torch.sqrt(var_y))

        return cka_score

    def calculate_cka_Matrix(self, activation):
        cka_metircs = []

        for idx, (key, output) in enumerate(activation.items()):

            li = []

            A = output
            if len(A.shape) == 3:
                (d1, d2, d3) = A.shape
                A = A.reshape(d1 * d2, d3)

            for idx2, (key, output2) in enumerate(activation.items()):
                B = output2

                if len(B.shape) == 3:
                    (d1, d2, d3) = B.shape
                    B = B.reshape(d1 * d2, d3)
                li.append(self.linear_CKA(A, B) * 100)

            cka_metircs.append(li.copy())

            reversed_arr = cka_metircs[::-1]
        return reversed_arr

    def register_hook(self):
        for full_name, module in self.model.encoder.named_modules():
            if module.__class__.__name__ in self.cka_analysis_layer_name_list:

                class_name = ""

                if module.__class__.__name__ == "T5LayerSelfAttention":
                    class_name = "EncoderSelfAttention"
                elif module.__class__.__name__ == "T5LayerFF":
                    class_name = "EncoderFeedForward"
                elif module.__class__.__name__ == "T5Block":
                    class_name = "EncoderBlock"
                else:
                    continue

                re_name = re.sub(r"\.(\d+)", r"[\1]", full_name)

                block_idx = re.search(r"block\[(\d+)\]", re_name).group(1)

                full_var_name = "self.model.encoder." + re_name
                self.model_name_list_enc.append(full_var_name)
                regist_hook_cmd = (
                    full_var_name
                    + ".register_forward_hook(self.get_activation_enc(f'"
                    + class_name
                    + "{"
                    + block_idx
                    + "}'))"
                )

                eval(regist_hook_cmd)

        for full_name, module in self.model.decoder.named_modules():
            if module.__class__.__name__ in self.cka_analysis_layer_name_list:

                class_name = ""

                if module.__class__.__name__ == "T5LayerSelfAttention":
                    class_name = "DecoderSelfAttention"
                elif module.__class__.__name__ == "T5LayerFF":
                    class_name = "DecoderFeedForward"
                else:
                    class_name = "DecoderCrossAttention"

                re_name = re.sub(r"\.(\d+)", r"[\1]", full_name)

                block_idx = re.search(r"block\[(\d+)\]", re_name).group(1)

                full_var_name = "self.model.decoder." + re_name
                self.model_name_list_dec.append(full_var_name)
                regist_hook_cmd = (
                    full_var_name
                    + ".register_forward_hook(self.get_activation_dec(f'"
                    + class_name
                    + "{"
                    + block_idx
                    + "}'))"
                )

                eval(regist_hook_cmd)

    def show_cka_matrix(self, cka_matrix, min_score=90, model_name=""):
        plt.imshow(cka_matrix, cmap="gray", vmin=min_score, vmax=100)
        plt.title(model_name + "output similarity")

        plt.colorbar()

        plt.axis("off")

        plt.show()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{model_name}_{current_time}.png"
        plt.savefig(filename)
        plt.close()

    def get_layer_modules(self, cka_scores, threshold):
        if threshold > 100:
            print("threshold should be under 100")
            return
        modules = []
        module_start_idx = 0

        temp_module = []

        start_idx = 0
        end_idx = 0

        while end_idx < len(cka_scores):
            if cka_scores[start_idx][end_idx] >= threshold:
                temp_module.append(end_idx)
            else:
                modules.append(temp_module.copy())
                start_idx = end_idx
                temp_module.clear()
                continue
            end_idx += 1

        if len(temp_module) != 0:
            modules.append(temp_module.copy())

        return modules

    def cka_analysis(self, threshold):
        self.enc_cka_matrix = torch.zeros(
            self.enc_activation_size, self.enc_activation_size
        ).to(self.device)
        self.dec_cka_matrix = torch.zeros(
            self.dec_activation_size, self.dec_activation_size
        ).to(self.device)
        enc_idx = 0
        dec_idx = 0
        enc_error_idx = []
        dec_error_idx = []

        for index, sample in enumerate(tqdm(self.test_dataset)):

            answer = sample["answers"]["text"][0]
            context = sample["context"]

            input_text = "<answer> %s <context> %s " % (answer, context)
            encoding = self.tokenizer.encode_plus(input_text, return_tensors="pt").to(
                self.device
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=3,
                    num_return_sequences=1,
                )

            new_enc_cka_matrix = torch.tensor(
                self.calculate_cka_Matrix(self.activation_enc), device="cuda"
            )
            if not torch.isnan(new_enc_cka_matrix).any():
                self.enc_cka_matrix = (
                    self.enc_cka_matrix * enc_idx + new_enc_cka_matrix
                ) / (enc_idx + 1)
                enc_idx += 1
            else:
                print(new_enc_cka_matrix)
                enc_error_idx.append(enc_idx)

            new_dec_cka_matrix = torch.tensor(
                self.calculate_cka_Matrix(self.activation_dec), device="cuda"
            )
            if not torch.isnan(new_dec_cka_matrix).any():
                self.dec_cka_matrix = (
                    self.dec_cka_matrix * dec_idx + new_dec_cka_matrix
                ) / (dec_idx + 1)
                dec_idx += 1

            else:
                print(new_dec_cka_matrix)
                dec_error_idx.append(dec_idx)

        self.show_cka_matrix(
            self.enc_cka_matrix.to("cpu"),
            min_score=threshold,
            model_name="encoder_cka_matrix",
        )
        self.show_cka_matrix(
            self.dec_cka_matrix.to("cpu"),
            min_score=threshold,
            model_name="decoder_cka_matrix",
        )

        enc_cka_matrix_for_make_moduel = self.enc_cka_matrix.cpu().numpy()[::-1]
        dec_cka_matrix_for_make_moduel = self.dec_cka_matrix.cpu().numpy()[::-1]

        self.enc_module_idx_list = self.get_layer_modules(
            enc_cka_matrix_for_make_moduel, threshold
        )
        self.dec_module_idx_list = self.get_layer_modules(
            dec_cka_matrix_for_make_moduel, threshold
        )

        print("encoder module:", self.enc_module_idx_list)
        print("decoder module:", self.dec_module_idx_list)
        print("encoder error data idx:", enc_error_idx)
        print("decoder error data idx", dec_error_idx)

    def layer_reduction(self, enc_module_list=None, dec_module_list=None):

        if enc_module_list is not None:
            self.enc_module_idx_list = enc_module_list

        if dec_module_list is not None:
            self.dec_module_idx_list = dec_module_list

        for items in self.enc_module_idx_list:
            temp = []
            for item in items:
                temp.append(self.model_name_list_enc[item])
            self.enc_module_layer_list.append(temp)

        for items in self.dec_module_idx_list:
            temp = []
            for item in items:
                temp.append(self.model_name_list_dec[item])
            self.dec_module_layer_list.append(temp)

        enc_delete_layer_list = [
            list(reversed(sub_arr))[:-1]
            for sub_arr in reversed(self.enc_module_layer_list)
            if len(sub_arr) > 1
        ]
        enc_delete_layer_list_flatten = [
            item for sublist in enc_delete_layer_list for item in sublist
        ]

        dec_delete_layer_list = [
            list(reversed(sub_arr))[:-1]
            for sub_arr in reversed(self.dec_module_layer_list)
            if len(sub_arr) > 1
        ]
        dec_delete_layer_list_flatten = [
            item for sublist in dec_delete_layer_list for item in sublist
        ]

        for item in enc_delete_layer_list_flatten:
            del_cmd = "del " + item
            exec(del_cmd)

        for item in dec_delete_layer_list_flatten:
            del_cmd = "del " + item
            exec(del_cmd)

    def train(self):

        self.model.train()

        for layer in self.model.modules():

            forward_hook_keys = list(layer._forward_hooks.keys())

            for key in forward_hook_keys:
                del layer._forward_hooks[key]

        start_time = time.time()
        pl.seed_everything(99)

        print("Using device:", self.device)
        print("Preparing dataset...")
        train_dataset = QGDataset(self.tokenizer, train_file_path)
        validation_dataset = QGDataset(self.tokenizer, validation_file_path)

        print("train_dataset: ", len(train_dataset))
        print("validation_dataset: ", len(validation_dataset))

        print("Initializing model...")
        self.model = T5FineTuner(
            self.model, self.tokenizer, train_dataset, validation_dataset, self.args
        )

        trainer = pl.Trainer(
            max_epochs=3,
            gpus=1,
            progress_bar_refresh_rate=30,
            callbacks=[EarlyStopping(monitor="val_loss")],
        )

        print("Run learning rate finder...")
        lr_finder = trainer.tuner.lr_find(self.model)
        print("Suggested lr: ", lr_finder.suggestion())

        print("Fine tuning...")
        trainer.fit(self.model)

        print("Saving model...")

        directory, filename = os.path.split(self.mpruned_model_path)

        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.model.model, self.mpruned_model_path)

        if not os.path.exists(self.tokenizer_path):
            os.makedirs(self.tokenizer_path)

        self.tokenizer.save_pretrained(self.tokenizer_path)

        end_time = time.time() - start_time
        print("Total time: %s hours" % (end_time / 60 / 60))

        ### eval code

        print("Loading metrics...")
        bleu = load_metric("bleu")
        rouge = load_metric("rouge")
        meteor = load_metric("meteor")

        QG = QuestionGeneration(self.mpruned_model_path, self.tokenizer_path)
        SE = SentenceEmbeddings()

        references = []
        predictions = []

        valid_dataset = load_dataset("squad", split="validation")

        for d in tqdm(valid_dataset):
            answer = d["answers"]["text"][0]
            question = d["question"]
            context = d["context"]

            references.append(question)

            qa_pair_list = QG.generate(answer, context)
            generated_question = SE.get_most_similar(context, qa_pair_list)
            predictions.append(generated_question["question"])

        print("Compute bleu...")
        bleu_references = [[word_tokenize(r)] for r in tqdm(references)]
        bleu_predictions = [word_tokenize(r) for r in tqdm(predictions)]
        results = bleu.compute(predictions=bleu_predictions, references=bleu_references)
        print(results)

        print("Compute rouge...")
        results = rouge.compute(predictions=predictions, references=references)
        print(results)

        print("Compute meteor...")
        results = meteor.compute(predictions=predictions, references=references)
        print(results)

        print("All done.")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="T5_Mpruner args")

    parser.add_argument("--num_workers", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--model_path", default="t5-base")

    parser.add_argument("--tokenizer_path", default="t5-base")

    parser.add_argument("--useHGModel", type=str2bool, default=False)

    parser.add_argument("--mpruned_model_path", default="mpruned.pth")

    args = parser.parse_args()

    t = T5_MPruner(
        args.model_path,
        args.tokenizer_path,
        args.mpruned_model_path,
        args=args,
        useHGModel=args.useHGModel,
    )

    t.tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<answer>", "<context>"]}
    )

    total_params = sum(p.numel() for p in t.model.parameters())
    print(f"Total number of parameters: {total_params}")
    print(
        "################################################################################################"
    )

    print("register hook")
    t.register_hook()
    print("cka_analysis")
    t.cka_analysis(99)

    t.layer_reduction()

    total_params = sum(p.numel() for p in t.model.parameters())
    print(f"Total number of parameters: {total_params}")

    t.train()
