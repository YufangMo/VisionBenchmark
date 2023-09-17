import torch
import torch.nn.functional as F
# import torch.cuda.amp as amp
from spikingjelly.clock_driven.functional import reset_net
import numpy as np
import random as python_random
import os
import json
import sys
sys.path.append('./model')
from models import snn_vgg, snn_resnet
from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    np.random.seed(seed)
    
    python_random.seed(seed)
    # cuda env
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    print('seeds are fixed')

def net_model(model_name, in_features, n_classes, timestep=5):
    if model_name == 'resnet19':
        return snn_resnet.ResNet19(in_features=in_features, num_classes=n_classes, total_timestep=timestep)
    elif model_name == 'vgg16':
        return snn_vgg.vgg16_bn(in_features=in_features, num_classes=n_classes, total_timestep=timestep)

def train(model, train_loader, criterion, optimizer, scheduler=None):
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        train_loss = 0.0

        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()
        output_list = model(imgs)
        output = sum(output_list) / len(output_list)
        train_loss = criterion(output, targets)
        train_loss.backward()
        optimizer.step()
        reset_net(model)
    if scheduler is not None:
        scheduler.step()
    return train_loss.item()

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data) # Output is a list
            output = sum(output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def dump_json(obj, fdir, name):
    """
    Dump python object in json
    """
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)

def load_json(fdir, name):
    """
    Load json as python object
    """
    path = os.path.join(fdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj 