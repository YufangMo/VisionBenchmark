from dataloaders.StaticDataloaders.dataloader import dataset
from utils_train import train, test, setup_seed
from models.RGBModel.snn_resnet import ResNet19
from models.BlackModel.snn_resnet import ResNet19 as C1_Resnet19
from config import get_args

from tqdm import tqdm
import torch.nn as nn
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

def main():
    args = get_args()
    trainset, valset, n_class = dataset(args.dataset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, pin_memory=True, num_workers=4,
                                            worker_init_fn=lambda k: setup_seed(args.seed + k + 1))
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, pin_memory=True, num_workers=4,
                                            worker_init_fn=lambda k: setup_seed(args.seed + k + 1)) 
    model = ResNet19(num_classes=n_class, total_timestep=args.timestep).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs),
                                                                       eta_min= 0)
    
    epochs = args.epochs
    best_acc = .0
    for epoch in range(epochs):
        loss = train(train_loader, model, criterion, optimizer, scheduler)
        acc = test(model, val_loader)
        if acc > best_acc:
            best_acc = acc
        print('epoch: ', epoch, 'loss: ', loss, 'acc: ', acc, 'best_acc: ', best_acc)

main()