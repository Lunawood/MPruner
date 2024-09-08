from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import gc
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch.nn as nn
import argparse
import copy
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
from transformers import AutoTokenizer


def show_cka_metrics(cka_metrics, min_score, save_path):
    plt.imshow(cka_metrics, cmap="gray", vmin=min_score, vmax=100)
    plt.title("output similarity")

    plt.colorbar()

    plt.axis("off")

    plt.savefig(save_path + "/cka_metrics.png")


def centering(K):
    n = K.shape[0]
    unit = torch.ones(n, n, device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return H @ K @ H


def HSIC(Kx, Ky):
    return torch.trace(Kx @ Ky)


def linear_CKA(X, Y):
    Kx = X @ X.T
    Ky = Y @ Y.T
    Kx_centered = centering(Kx)
    Ky_centered = centering(Ky)
    hsic = HSIC(Kx_centered, Ky_centered)
    var_x = HSIC(Kx_centered, Kx_centered)
    var_y = HSIC(Ky_centered, Ky_centered)
    cka_score = hsic / torch.sqrt(var_x * var_y)
    return cka_score.item()


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output[0]

    return hook


def register_hooks(model, hook):
    for i in range(len(model.bert.encoder.layer)):
        othername = hook + str(i)
        model.bert.encoder.layer[i].register_forward_hook(get_activation(othername))
        layer_name[i] = "model.bert.encoder.layer[" + str(i) + "]"


def calculate_cka_Metrics(layer_prefix):
    cka_metircs = []
    for layer_i in range(len(activation)):
        layer_name_i = layer_prefix + str(layer_i)

        li = []

        A = activation[layer_name_i][0]

        for compare_layer_index in range(len(activation)):
            compare_layer_name = layer_prefix + str(compare_layer_index)
            B = activation[compare_layer_name][0]
            li.append(linear_CKA(A, B) * 100)

        cka_metircs.append(li.copy())
        reversed_arr = cka_metircs[::-1]

    return np.array(reversed_arr)


def get_layer_modules(cka_scores, threshold):
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


def Model(useHGModel, load):
    if useHGModel:
        tokenizer = AutoTokenizer.from_pretrained(load)
        model = AutoModelForSequenceClassification.from_pretrained(load)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"{load}/tokenizer")
        model = torch.load(f"{load}/bert_original.pth")

    model.to(device)

    return model, tokenizer


def Dataset(tokenizer, data_name):
    dataset = load_dataset(data_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = load_dataset("emotion")
    train_dataset, val_dataset, test_dataset = (
        dataset["train"],
        dataset["validation"],
        dataset["test"],
    )

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    split_test = tokenized_test_dataset.train_test_split(test_size=0.5)

    test_data = split_test["train"]
    valid_data = split_test["test"]

    return tokenized_train_dataset, valid_data, test_data


def GetAccuracy(trainer):
    return trainer.evaluate()["eval_accuracy"]


def GetCandidates(model, tokenizer, hook, threshold, test_loader, save_path):
    register_hooks(model, hook)
    length = len(model.bert.encoder.layer)
    cka_metrics = torch.zeros(length, length)
    iter = 1

    model.eval()
    for _ in range(iter):
        for index, data in enumerate(tqdm(test_loader)):
            text = data["text"]
            try:
                inputs = tokenizer(text, return_tensors="pt")
                inputs = inputs.to(device)
            except:
                continue
            with torch.no_grad():
                outputs = model(**inputs)

            new_calculate_cka_Metrics = calculate_cka_Metrics(hook)
            isnan_arr = np.isnan(new_calculate_cka_Metrics)

            if np.any(isnan_arr):
                continue

            cka_metrics = (cka_metrics * index + new_calculate_cka_Metrics) / (
                index + 1
            )

    show_cka_metrics(cka_metrics, threshold, save_path)
    cka_metrics_to_numpy = cka_metrics.numpy()
    cka_metrics_to_numpy = cka_metrics_to_numpy[::-1]
    modules = get_layer_modules(cka_metrics_to_numpy, threshold)

    return modules


def CheckCluster(clusters):
    for cluster in clusters:
        if len(cluster) != 1:
            return False
    return True


def Pruner(model, cluster, k):
    nofreeze_layer = set()

    for _, i in enumerate(cluster[::-1]):
        for j in i[1::k][::-1]:
            del_cmd = "del " + str(layer_name[j])
            exec(del_cmd)
            nofreeze_layer = {x - 1 for x in nofreeze_layer}
            nofreeze_layer.add(j)
            nofreeze_layer.add(j - 1)

    return model, nofreeze_layer


def Freezer(model, unfreeze_list, hook, freeze_flag):
    if freeze_flag:
        for name, module in model.named_modules():
            if hook == module.__class__.__name__:
                code = (
                    "model."
                    + re.sub(r"\.(\d+)", r"[\1]", name)
                    + ".requires_grad = False"
                )
                exec(code)

    index = 0

    for name, module in model.named_modules():
        if hook == module.__class__.__name__:
            if index in unfreeze_list:
                code = (
                    "model."
                    + re.sub(r"\.(\d+)", r"[\1]", name)
                    + ".requires_grad = True"
                )
                exec(code)
            index += 1

    return model


def Save(model, save_path):
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()

    torch.save(model, save_path + "/mpruner_model.pth")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


def CreateTrainer(model, tokenizer, train_loader, val_loader, epoch):
    model.to(device)

    training_args = TrainingArguments(
        output_dir=f"./model_save",
        num_train_epochs=epoch,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.001,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        compute_metrics=compute_metrics,
    )

    return trainer


def Emotion_Mpruner(
    useHGModel,
    load,
    data_name,
    hook,
    threshold,
    acc_threshold,
    epoch,
    iterate,
    freeze_flag,
    save_path,
):
    # Model load
    model, tokenizer = Model(useHGModel, load)
    new_model = copy.deepcopy(model)
    model.to(device)

    train_loader, val_loader, test_loader = Dataset(tokenizer, data_name)

    # clusters init and clear
    clusters = []

    # calculate accuarcy
    trainer = CreateTrainer(model, tokenizer, train_loader, val_loader, epoch)
    acc_origin = GetAccuracy(trainer)

    k = iterate

    while k <= 3:
        clusters = GetCandidates(
            model, tokenizer, hook, threshold, test_loader, save_path
        )
        print(clusters)
        if CheckCluster(clusters) == True:
            break

        model, unfreeze_list = Pruner(model, clusters, k)
        model = Freezer(model, unfreeze_list, hook, freeze_flag)

        # Model clear hook
        for module in model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
            module._backward_hooks.clear()

        activation.clear()
        layer_name.clear()

        model.train()
        trainer = CreateTrainer(model, tokenizer, train_loader, val_loader, epoch)
        trainer.train()

        model = trainer.model

        model.eval()
        acc_pruner = GetAccuracy(trainer)

        print("acc_pruner: ", acc_pruner)

        if acc_origin - acc_pruner <= acc_threshold:
            new_model = copy.deepcopy(model)
        else:
            model = copy.deepcopy(new_model)
            k += 1

    Save(model, save_path)
    print("acc_origin: ", acc_origin)
    print("acc_pruner: ", acc_pruner)
    print("model parameter: ", model.num_parameters() / 1000000, "M")


if __name__ == "__main__":
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--useHGModel", type=bool, default=True)
    parser.add_argument(
        "--load", type=str, default="bhadresh-savani/bert-base-uncased-emotion"
    )
    parser.add_argument("--data_name", type=str, default="emotion")
    parser.add_argument("--hook", type=str, default="BertLayer")
    parser.add_argument("--threshold", type=int, default=98)
    parser.add_argument("--acc_threshold", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--iterate", type=int, default=1)
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--save", type=str, default="./result")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activation = {}
    layer_name = {}

    Emotion_Mpruner(
        args.useHGModel,
        args.load,
        args.data_name,
        args.hook,
        args.threshold,
        args.acc_threshold,
        args.epoch,
        args.iterate,
        args.freeze,
        args.save,
    )
