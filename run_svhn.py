import torch
from flaml import tune
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import SVHN
from torchvision.transforms import transforms
from logging_utils.tbtools import AverageMeter
import copy
import math
import loss_utils
import models
import os
import argparse

svhn_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("--init", type=int, default=3990, help="initial coreset size")
    arg.add_argument("--limit", type=int, default=4100, help="upper limit of coreset size")
    arg.add_argument("--tolerance", type=str, default="20%")
    arg.add_argument("--dataset_size", type=int, default=73257)
    arg.add_argument("--train_epoch", type=int, default=100)
    arg.add_argument("--epoch_converge", type=int, default=100)
    arg.add_argument("--data", type=str, default="dataset")
    arg.add_argument("--save_dir", type=str, default="configs")
    input_args = arg.parse_args()
    return input_args

args = parse_args()

def get_svhn_train_loader(batch_size=1024):
    train_dataset = SVHN(args.data, split='train', transform=svhn_transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=5)
    return train_loader, train_dataset

def get_svhn_test_loader():
    test_loader = torch.utils.data.DataLoader(SVHN(args.data, split='test', transform=svhn_transform), batch_size=1024, pin_memory=True, num_workers=4)
    return test_loader

def get_all_data(full_train_loader):
    datas, targets = [], []
    for data, target in full_train_loader:
        datas.append(data)
        targets.append(target)
    return torch.cat(datas), torch.cat(targets)

def get_loss_on_full_train(model, full_loader):
    model.eval()
    loss_avg = 0

    for X, Y in full_loader:
        data, target = X.cuda(), Y.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction="sum")
        loss_avg += loss.item()

    loss_avg /= len(full_loader.dataset)
    return loss_avg


def train(net, trainloader, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    correct /= len(trainloader.dataset)
    return correct, train_loss

def test(model, test_loader):
    loss, acc1 = 0, 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss += F.cross_entropy(output, target).item()
            _, pred = output.max(1)
            acc1 += pred.eq(target).sum().item()

    acc1 /= len(test_loader.dataset)
    return acc1, loss

def train_to_converge(model, x_coreset, y_coreset, epoch_converge=100):
    model_copy = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=0.001)
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    data, target = x_coreset.cuda(), y_coreset.cuda()

    diverged = False
    for i in range(epoch_converge):
        optimizer.zero_grad()
        output = model_copy(data)
        acc1, acc5 = loss_utils.accuracy(output, target, topk=(1, 5))
        loss = F.cross_entropy(output, target)
        losses.update(loss.item(), target.size(0))
        top1.update(acc1, target.size(0))
        top5.update(acc5, target.size(0))
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"{i}th iter, inner loss {loss.item()}")
    if math.isnan(loss.item()) or loss > 8:
        diverged = True
    return model_copy, diverged

epoch = 0

def evaluate_function(config):
    model = models.SVHNCnnMini().cuda()
    global epoch
    epoch += 1

    masks = []
    for i in range(args.dataset_size):
        masks.append(config[f"{i}"])
    masks = torch.tensor(masks)
    masks = (masks>0).int()
    indices = torch.nonzero(masks).squeeze()
    coreset_size = indices.shape[0]

    train_loader_no_transform, _ = get_svhn_train_loader()

    X, Y = get_all_data(train_loader_no_transform)
    x_coreset, y_coreset = X[indices], Y[indices]

    if coreset_size >= args.limit:
        loss_on_full_train = 1000 + 1000 * (coreset_size - args.limit) ** 2
        print("Coreset size exceeded, size: ", coreset_size)
        return {"loss": loss_on_full_train, "size": coreset_size}

    model_copy_converged, diverged = train_to_converge(model, x_coreset, y_coreset, epoch_converge=args.epoch_converge)
    loss_on_full_train = get_loss_on_full_train(model_copy_converged, train_loader_no_transform)

    if diverged:
        loss_on_full_train = 10000
    del X, Y, x_coreset, y_coreset, model_copy_converged

    print("epoch: ", epoch, "Loss on full data: ", loss_on_full_train, "Coreset size: ", coreset_size)
    return {"loss": loss_on_full_train, "size": coreset_size}

def optimize():
    initial_config = dict()
    search_space = dict()
    np.random.seed(42)

    # define search space
    for i in range(args.dataset_size):
        search_space[f"{i}"] = tune.uniform(lower=-1, upper=1)

    # set some random mask to 1
    indexes = np.random.choice(args.dataset_size, args.init, replace=False)
    for i in range(args.dataset_size):
        if i in indexes:
            initial_config[f"{i}"] = 0.3
        else:
            initial_config[f"{i}"] = -0.5

    objectives = dict()
    objectives["metrics"] = ["loss", "size"]
    objectives["tolerances"] = {"loss": args.tolerance, "size": "0%"}
    objectives["targets"] = {"loss": 0.0, "size": 0.0}
    objectives["modes"] = ["min", "min"]

    analysis = tune.run(
        evaluate_function,
        num_samples=300,
        time_budget_s=99999999,
        config=search_space,
        use_ray=False,
        lexico_objectives=objectives,
        low_cost_partial_config=initial_config,
        max_failure=400,
        verbose=0,
        step=8
    )

    return analysis

def evaluate_results(analysis):
    config = analysis.get_best_trial().config

    masks = []
    for i in range(args.dataset_size):
        masks.append(config[f"{i}"])
    masks = torch.tensor(masks)
    masks = (masks > 0).int()
    indices = torch.nonzero(masks).squeeze()

    if not os.path.exists(args.save_dir) or not os.path.exists(os.path.join(args.save_dir, "svhn")):
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "svhn"), exist_ok=True)

    torch.save(indices, os.path.join(args.save_dir, f"svhn/{args.limit}.pt"))

    full_train_loader, full_dataset = get_svhn_train_loader()
    subset = torch.utils.data.Subset(full_dataset, indices=indices)
    trainloader = torch.utils.data.DataLoader(subset, batch_size=128, num_workers=7, shuffle=True, pin_memory=True)

    print("Coreset size: ", analysis.best_result["size"])

    test_loader = get_svhn_test_loader()
    acc_mean = []

    for i in range(5):
        model_train = models.SVHNCnnModel().cuda()
        optimizer = torch.optim.Adam(model_train.parameters(), lr=0.001)
        best_acc1, best_train_acc1 = 0, 0

        for epoch in range(0, args.train_epoch):
            train_acc1, train_loss = train(model_train, trainloader, optimizer)
            test_acc1, test_loss = test(model_train, test_loader)
            best_acc1 = max(test_acc1, best_acc1)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            if (epoch + 1) % 50 == 0 or epoch == args.train_epoch - 1:
                print(f"epoch {epoch}, train acc1 {train_acc1}, train loss {train_loss}")
                print(f"epoch {epoch}, test acc1 {test_acc1}, test loss {test_loss}")
                print(f"best acc1: {best_acc1}, best train acc1: {best_train_acc1}")

        acc_mean.append(best_acc1)
    print(acc_mean)

if __name__ == "__main__":
    analysis = optimize()
    evaluate_results(analysis)
