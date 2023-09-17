import argparse


def get_args():
    parser = argparse.ArgumentParser("SNN-LTH")
    parser.add_argument('--exp_name', type=str, default='snn_pruning',  help='experiment name')
    parser.add_argument('--data_dir', type=str, default='/home/yuhong/Projects/Data/', help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='mnist', help='[cifar10, cifar100, svhn, fmnist, mnist]')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model_name', type=str, default='vgg16', help='[resnet19, vgg16]')

    parser.add_argument('--timestep', type=int, default=1, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default= 128, help='batch size')

    parser.add_argument('--optimizer', type=str, default='sgd', help='[sgd, adam, adamw, adan]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine]')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='learnng rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', default=3, type=int)

    args = parser.parse_args()
    print(args)

    return args