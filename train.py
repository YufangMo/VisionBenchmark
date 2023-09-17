from dataloaders.dataloader import dataset
import utils
from config import get_args
from tqdm import tqdm
import torch.nn as nn
import torch

def main():
    args = get_args()
    utils.set_seed(args.seed)
    trainset, valset, in_features, n_classes = dataset(args.dataset, data_dir=args.data_dir)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, pin_memory=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, pin_memory=True, num_workers=4) 
    model = utils.net_model(args.model_name, in_features, n_classes, args.timestep).cuda()
    criterion = nn.CrossEntropyLoss()
    
    # The optimizers are to be set
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs),
                                                                       eta_min= 0)
    
    epochs = args.epochs
    best_acc = .0
    acc_list = []
    loss_list = []
    for epoch in range(epochs):
        loss = utils.train(model, train_loader, criterion, optimizer, scheduler)
        acc = utils.test(model, val_loader)
        if acc > best_acc:
            best_acc = acc
        print(f'epoch:{epoch}, loss:{loss:.6f}, acc:{acc:.2f}, best_acc:{best_acc:.2f}')
        
        acc_list.append(acc)
        loss_list.append(loss)
    
    # Store the Result
    args_dict = vars(args)
    args_dict['acc_list'] = acc_list
    args_dict['loss_list'] = loss_list
    file_dir = './Exp_Result'
    file_name = f'optim{args.optimizer}_lr{args.learning_rate}_weightdecay{args.weight_decay}.pkl'
    utils.dump_json(args_dict, file_dir, file_name)

if __name__ == '__main__':
    main()