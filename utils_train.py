import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
from spikingjelly.clock_driven.functional import reset_net
import numpy as np

def train(train_data,  model, criterion, optimizer, scheduler):
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_data):
        train_loss = 0.0

        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()
        output_list = model(imgs)
        output = sum(output_list) / len(output_list)
        train_loss = criterion(output, targets)
        train_loss.backward()
        optimizer.step()
        reset_net(model)
    scheduler.step()
    return train_loss.item()

def train_prune(train_data,  model, criterion, optimizer, scheduler):
    model.train()
    EPS = 1e-6
    for batch_idx, (imgs, targets) in enumerate(train_data):
        train_loss = 0.0
        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()
        output_list = model(imgs)
        output = sum(output_list) / len(output_list)
        train_loss = criterion(output, targets)
        train_loss.backward()
        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data
                if (len(tensor.size())) == 1:
                    continue
                grad_tensor = p.grad
                grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                p.grad.data = grad_tensor
        optimizer.step()
        reset_net(model)
    scheduler.step()
    return train_loss.item()

def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) # Output is a list
            output = sum(output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def test_ann(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def setup_seed(seed):
     torch.manual_seed(seed)
     np.random.seed(seed)