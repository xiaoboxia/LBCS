import torch
import torch.nn.functional as F
from datasets_utils.cifar10 import CIFAR10
from models import ResNet18
from torchvision.transforms import transforms
import argparse


def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("--data", default="dataset")
    input_args = arg.parse_args()
    return input_args

args = parse_args()

cifar_transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def get_cifar_train_loader(batch_size=512):
    train_dataset = CIFAR10(root=args.data, train=True, transform=cifar_transform_train, download=True)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args["num_worker"])
    return loader, train_dataset

def get_cifar_test_loader(batch_size=500):
    dataset = CIFAR10(root=args.data, train=False, transform=cifar_transform_test, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args["num_worker"])

    return loader

def train(net, trainloader, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, _) in enumerate(trainloader):
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
        for data, target, _ in test_loader:
            # import pdb; pdb.set_trace()
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss += F.cross_entropy(output, target).item()
            _, pred = output.max(1)
            acc1 += pred.eq(target).sum().item()

    acc1 /= len(test_loader.dataset)
    return acc1, loss

def evaluate_results():    
    trainloader, _ = get_cifar_train_loader(batch_size=128)
    
    test_loader = get_cifar_test_loader()

    for i in range(20):
        print("Obtaining checkpoint", i)
        model_train = ResNet18().cuda()
        optimizer = torch.optim.SGD(model_train.parameters(), momentum=0.9, lr=0.1, weight_decay=5e-4)
        best_acc1, best_train_acc1 = 0, 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        for epoch in range(0, 180):
            train_acc1, train_loss = train(model_train, trainloader, optimizer)
            test_acc1, test_loss = test(model_train, test_loader)
            scheduler.step()
            best_acc1 = max(test_acc1, best_acc1)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            if epoch % 50 == 0 or epoch == 179:
                print(f"epoch {epoch}, train acc1 {train_acc1}, train loss {train_loss}")
                print(f"epoch {epoch}, test acc1 {test_acc1}, test loss {test_loss}")
                print(f"best test acc1: {best_acc1}, best train acc1: {best_train_acc1}")
        torch.save(model_train.state_dict(), f"checkpoint/{i}.pt")

if __name__ == "__main__":
    analysis = evaluate_results()
