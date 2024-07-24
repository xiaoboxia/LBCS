import torch
from flaml import tune
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from logging_utils.tbtools import AverageMeter
import math
import loss_utils
import models
import argparse
import os

def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("--init", type=int, default=2000, help="initial coreset size")
    arg.add_argument("--limit", type=int, default=2200, help="upper limit of coreset size")
    arg.add_argument('--noise_rate', default=0.3, type=float)
    arg.add_argument('--epoch_converge', default=20, type=int)
    arg.add_argument('--dataset_size', default=60000, type=int)
    arg.add_argument("--noisify", default=False, action="store_true")
    arg.add_argument("--data", type=str, default="dataset")
    arg.add_argument('--train_epoch', default=100, type=int)
    arg.add_argument("--tolerance", type=str, default="15%")
    arg.add_argument("--imbalance", default=False, action="store_true")
    arg.add_argument("--save_dir", type=str, default="configs")

    input_args = arg.parse_args()
    return input_args

args = parse_args()

def get_fashion_train_loader(batch_size=128):
    train_dataset = FashionMNIST(args.data, train=True, transform=transforms.ToTensor(), download=True)
    if args.noisify:
        train_dataset.targets = np.load(f"tmp/fashion_noisy_target.npy")
        train_dataset.targets = torch.from_numpy(train_dataset.targets)
    if args.imbalance:
        train_dataset.targets = torch.from_numpy(np.load(f"tmp/fashion_imb_targets.npy"))
        train_dataset.data = torch.from_numpy(np.load(f"tmp/fashion_imb_train.npy"))    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=5)
    return train_loader, train_dataset


def get_all_data(full_train_loader):
    datas, targets = [], []
    for data, target in full_train_loader:
        datas.append(data)
        targets.append(target)
    return torch.cat(datas), torch.cat(targets)

loader, _ = get_fashion_train_loader()
x, y = get_all_data(loader)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    data, target = x_coreset.cuda(), y_coreset.cuda()
    
    diverged = False
    for i in range(epoch_converge):
        optimizer.zero_grad()
        output = model(data)
        acc1, acc5 = loss_utils.accuracy(output, target, topk=(1, 5))
        loss = F.cross_entropy(output, target)
        losses.update(loss.item(), target.size(0))
        top1.update(acc1, target.size(0))
        top5.update(acc5, target.size(0))
        loss.backward()
        optimizer.step()
    if math.isnan(loss.item()) or loss > 8:
        diverged = True
    print("Top1: ", top1)
    return diverged
epoch = 0

def evaluate_function(config):
    global epoch
    epoch += 1
    model = models.LeNet().cuda()

    masks = []
    for i in range(args.dataset_size):
        masks.append(config[f"{i}"])
    masks = torch.tensor(masks)
    masks = (masks>0).int()
    indices = torch.nonzero(masks).squeeze()
    coreset_size = indices.shape[0]
    
    train_loader_no_transform, _ = get_fashion_train_loader()

    x_coreset, y_coreset = x[indices], y[indices]

    if coreset_size >= args.limit:
        loss_on_full_train = 1000 + 1000 * (coreset_size - args.limit) ** 2
        print("Coreset size exceeded, size: ", coreset_size)
        return {"loss": loss_on_full_train, "size": coreset_size}
    
    diverged = train_to_converge(model, x_coreset, y_coreset, epoch_converge=args.epoch_converge)
    loss_on_full_train = get_loss_on_full_train(model, train_loader_no_transform)

    if diverged:
        loss_on_full_train = 10000

    print("Epoch: ", epoch, "Loss on full data: ", loss_on_full_train, "Coreset size: ", coreset_size)
    return {"loss": loss_on_full_train, "size": coreset_size}

def optimize():
    initial_config = dict()
    search_space = dict()

    # define search space
    for i in range(args.dataset_size):
        search_space[f"{i}"] = tune.uniform(lower=-1, upper=1)

    # set some random mask to 1
    np.random.seed(41)
    indexes = np.random.choice(args.dataset_size, args.init, replace=False)
    for i in range(args.dataset_size):
        if i in indexes:
            initial_config[f"{i}"] = 0.2
        else:
            initial_config[f"{i}"] = -0.5

    objectives = dict()
    objectives["metrics"] = ["loss", "size"]
    objectives["tolerances"] = {"loss": args.tolerance, "size": "0%"}
    objectives["targets"] = {"loss": 0.0, "size": 0.0}
    objectives["modes"] = ["min", "min"]

    analysis = tune.run(
        evaluate_function,
        num_samples=500,
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
    masks = (masks>0).int()
    indices = torch.nonzero(masks).squeeze()
    coreset_size = indices.shape[0]
    print("Coreset size: ", coreset_size)
    
    if not os.path.exists(args.save_dir) or not os.path.exists(os.path.join(args.save_dir, "fashion")):
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "fashion"), exist_ok=True)

    torch.save(indices, os.path.join(args.save_dir, f"fashion/{args.limit}.pt"))

    full_dataset = FashionMNIST(args.data, train=True, transform=transforms.ToTensor())
    if args.imbalance:
        full_dataset.targets = torch.from_numpy(np.load(f"tmp/fashion_imb_targets.npy"))
        full_dataset.data = torch.from_numpy(np.load(f"tmp/fashion_imb_train.npy"))
    if args.noisify:
        full_dataset.targets = np.load(f"tmp/fashion_noisy_target.npy")
        full_dataset.targets = torch.from_numpy(full_dataset.targets)

    subset = torch.utils.data.Subset(full_dataset, indices=indices)
    trainloader = torch.utils.data.DataLoader(subset, batch_size=128, num_workers=7, shuffle=True, pin_memory=True)

    test_dataset = FashionMNIST(args.data, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=7, shuffle=True, pin_memory=True)
    acc_mean = []

    for i in range(5):
        model_train = models.LeNet().cuda()
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
