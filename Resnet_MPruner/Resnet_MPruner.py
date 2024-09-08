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


class CKA:
    def __init__(self, activation_size):
        self.hsic_accumulator = torch.zeros(activation_size, activation_size)

    def generate_gram_matrix(self, x):
        x = x.view(x.size(0), -1)
        gram = torch.matmul(x, x.t())

        n = gram.size(0)
        gram.fill_diagonal_(0)

        means = gram.sum(dim=0) / (n - 2)
        means -= means.sum() / (2 * (n - 1))

        gram -= means.unsqueeze(1)
        gram -= means.unsqueeze(0)

        gram.fill_diagonal_(0)

        gram = gram.view(-1)

        return gram

    def update_state(self, activations):
        layer_grams = [self.generate_gram_matrix(x).to("cpu") for x in activations]
        layer_grams = torch.stack(layer_grams, 0)
        self.hsic_accumulator += torch.matmul(layer_grams, layer_grams.t())

    def result(self):
        mean_hsic = self.hsic_accumulator
        normalization = torch.sqrt(torch.diag(mean_hsic))

        mean_hsic /= normalization.unsqueeze(1)
        mean_hsic /= normalization.unsqueeze(0)

        return mean_hsic * 100


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output

    return hook


def register_hooks(model, layer_hook):
    hooks = []

    idx = 0

    for name, module in model.named_modules():
        if layer_hook == module.__class__.__name__:
            othername = "layer_" + str(idx).zfill(5)
            hooks.append(module.register_forward_hook(get_activation(othername)))
            layer_name[idx] = "model." + re.sub(r"\.(\d+)", r"[\1]", name)
            idx += 1

    return hooks


def get_layer_modules2(cka_scores, threshold):
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
        model = ResNetForImageClassification.from_pretrained(load)
    else:
        model = torch.load(load)
    model.to(device)

    return model


def Dataset(custom_batch_size=64):
    transform_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        "./data/imagenet/train", transform=transform_val
    )
    val_dataset = datasets.ImageFolder("./data/imagenet/val", transform=transform_val)
    test_dataset = datasets.ImageFolder("./data/imagenet/test", transform=transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=custom_batch_size, shuffle=True, num_workers=32
    )
    val_loader = DataLoader(
        val_dataset, batch_size=custom_batch_size, shuffle=False, num_workers=32
    )
    test_loader = DataLoader(
        test_dataset, batch_size=custom_batch_size, shuffle=True, num_workers=32
    )

    return train_loader, val_loader, test_loader


def GetAccuracy(model, val_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            logitss = model(inputs).logits

            gc.collect()
            torch.cuda.empty_cache()

            for idx, logits in enumerate(logitss):
                predicted_label = logits.argmax(-1).item()
                ground_truth = labels[idx].item()
                if predicted_label == ground_truth:
                    correct += 1
                total += 1

    return (correct / total) * 100


def GetCandidates(model, hook, threshold, val_loader, batch_size, save_path):
    register_hooks(model, hook)

    model(torch.randn(batch_size, 3, 224, 224).to(device))
    cka = CKA(len(activation))

    with torch.no_grad():
        for inputs, _ in tqdm(val_loader):
            inputs = inputs.to(device)
            logitss = model(inputs).logits

            activation_sort = [activation[key] for key in sorted(activation.keys())]
            cka.update_state(activation_sort)
            gc.collect()
            torch.cuda.empty_cache()
        cka.result()

    heatmap = cka.result().detach().numpy()[::-1]

    plt.figure(figsize=(100, 100))
    plt.imshow(heatmap, cmap="gray", vmin=threshold, vmax=100)
    plt.title("resnet output similarity")

    plt.colorbar()

    plt.grid(True)

    x_ticks = np.arange(0, heatmap.shape[1], 1)
    plt.xticks(x_ticks)
    plt.yticks(x_ticks)

    plt.savefig(save_path + "/heatmap.png")

    heatmap = heatmap[::-1]
    cluster = get_layer_modules2(heatmap, threshold=threshold)

    return cluster


def CheckCluster(clusters):
    for cluster in clusters:
        if len(cluster) != 1:
            return False
    return True


def func(model, layer, in_channels_list=[], out_channels_list=[]):
    for _, child in layer.named_children():
        try:
            in_channels_list.append(child.in_channels)
            out_channels_list.append(child.out_channels)
        except:
            pass
        if list(child.children()):
            func(model, child, in_channels_list, out_channels_list)
    return [in_channels_list[0], out_channels_list[len(in_channels_list) - 1]]


def ChannelCheck(model, prev_layer, next_layer):
    if (
        func(model, eval(layer_name[prev_layer]))[1]
        == func(model, eval(layer_name[next_layer]))[0]
    ):
        return True
    else:
        False


def Pruner(model, cluster, k):
    nofreeze_layer = set()

    for i in cluster[::-1]:
        for j in i[1::k][::-1]:
            if ChannelCheck(model, j - 1, j + 1):
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


def Trainer(freeze_model, num_epoch, train_loader):

    learning_rate = 0.0002

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(freeze_model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        freeze_model.train()
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = freeze_model(inputs)
            loss = loss_func(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    return freeze_model


def Save(model, save_path):
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()

    torch.save(model, save_path + "/mpruner_model.pth")


def Resnet_Mpruner(
    useHGModel,
    load,
    batch_size,
    hook,
    threshold,
    acc_threshold,
    epoch,
    iterate,
    freeze_flag,
    save_path,
):
    # Model load
    model = Model(useHGModel, load)
    model.to(device)

    train_loader, val_loader, test_loader = Dataset(batch_size)

    new_model = copy.deepcopy(model)

    # clusters init and clear
    clusters = []

    # calculate accuarcy
    acc_origin = GetAccuracy(model, val_loader)

    k = iterate

    while k <= 3:
        clusters = GetCandidates(
            model, hook, threshold, test_loader, batch_size, save_path
        )

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

        model = Trainer(model, epoch, train_loader)
        model.eval()
        acc_pruner = GetAccuracy(model, val_loader)

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
        "--load", type=str, default="anonymous78784949/resnet50-imagenet1k"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hook", type=str, default="ResNetBottleNeckLayer")
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

    Resnet_Mpruner(
        args.useHGModel,
        args.load,
        args.batch_size,
        args.hook,
        args.threshold,
        args.acc_threshold,
        args.epoch,
        args.iterate,
        args.freeze,
        args.save,
    )
