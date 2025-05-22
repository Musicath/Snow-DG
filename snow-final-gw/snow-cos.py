import time
import torch
from torch import nn
import torchvision
from torchvision import transforms
from resnet import resnet18
from methods.whitening import Whitening2dZCA
from pprint import pprint


class args_class():
    def __init__(self):
        self.alpha = 1.0
        self.batch_size = 128
        self.lr = 0.01
        self.momentum = 0.9
        self.num_epochs = 100
        self.num_groups = None
        self.num_iterations = None
        self.num_workers = None
        self.seed = None
        self.weight_decay = 0.005
        self.weight_regular = None


def std(sample):  # 构造单位向量
    gamma = torch.sum(sample ** 2, dim=1)
    gamma = torch.sqrt(gamma) + 1e-8
    return torch.t(torch.t(sample) / gamma)


def std_fc(features, weight, std_regular=True):
    if std_regular:
        features = std(features)
        weight = std(weight)
    return torch.mm(features, torch.t(weight))


def load_data(batch_size, resize=None, root='../Dataset', num_workers=0):
    num_workers = num_workers
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = [transforms.Resize((resize, resize)), transforms.ToTensor(), normalize]
    transform = torchvision.transforms.Compose(trans)
    pacs_train = torchvision.datasets.ImageFolder(root=root, transform=transform)
    pacs_test = torchvision.datasets.ImageFolder(root=root, transform=transform)
    train_iter = torch.utils.data.DataLoader(pacs_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(pacs_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


def data_generate():
    train_data = []
    test_id = []
    test_ood = []
    i = 0
    for X, y in train_pac:
        lable = torch.zeros(len(y), 21)
        for k in range(len(y)):
            lable[k][y[k]] = 1
        if 5 * i < 4 * len(train_pac):
            train_data.append({'X': X, 'y': lable})
        else:
            test_id.append({'X': X, 'y': lable})
        i += 1
    for X, y in test_s:
        lable = torch.zeros(len(y), 21)
        for k in range(len(y)):
            lable[k][y[k]] = 1
        test_ood.append({'X': X, 'y': lable})

    return train_data, test_id, test_ood


def evaluate_accuracy(test_data, net, device, args):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for i in range(len(test_data)):
            X = test_data[i]['X']
            y = test_data[i]['y']
            net.eval()
            features = net(X.to(device))
            weight = torch.zeros(7, 512).to(device)
            snow_bw = args.weight_regular(net.snow)
            for k in range(7):
                weight[k] = snow_bw[k]
            y_hat = std_fc(features, weight, std_regular=True)
            acc_sum += ((y_hat.argmax(dim=1) % 7) == (y.to(device).argmax(dim=1) % 7)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
    return acc_sum / n


def train(net, batch_size, optimizer, device, args):
    pprint(args.__dict__)
    net = net.to(device)
    print("training on ", device)
    result = []
    for epoch in range(args.num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for i in range(len(train_data)):
            X = train_data[i]['X'].to(device)
            y = train_data[i]['y'].to(device)
            features = net(X)
            weight = torch.zeros(21, 512).cuda()
            snow_bw = args.weight_regular(net.snow)
            for k in range(21):
                weight[k] = snow_bw[k % 7] + args.alpha * snow_bw[k + 7]
            y_hat = std_fc(features, weight, std_regular=True)
            l = ((1 - y_hat) ** 2 * y).sum() / batch_size
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            n += y.shape[0]
            batch_count += 1
        train_acc = evaluate_accuracy(train_data, net, device, args)
        test_id_acc = evaluate_accuracy(test_id, net, device, args)
        test_ood_acc = evaluate_accuracy(test_ood, net, device, args)
        print('epoch %d, loss %.4f, train acc %.3f, test_id acc %.3f, test_ood acc %.3f, time %.1f sec' % (
            epoch + 1, train_l_sum / batch_count, train_acc, test_id_acc, test_ood_acc, time.time() - start))
        result.append([epoch + 1, train_acc, test_id_acc, test_ood_acc])
    result = list(result)
    result.sort(key=lambda x: (x[2], x[3]), reverse=True)
    print('best 3 epochs:')
    for i in range(3):
        print('epoch %d, train acc %.3f, test_id acc %.3f, test_ood acc %.3f' % (result[i][0], result[i][1], result[i][2], result[i][3]))
    return result[0]


if __name__ == '__main__':
    total_time = time.time()
    args = args_class()
    args.batch_size = 128
    args.num_workers = 4
    args.num_epochs = 1
    args.weight_regular = Whitening2dZCA(track_running_stats=False, axis=0, eps=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_root = '../Dataset/PACS_21/train'
    test_root = '../Dataset/PACS_21/test'
    train_pac, test_pac = load_data(args.batch_size, resize=224, root=train_root, num_workers=args.num_workers)
    _, test_s = load_data(args.batch_size, resize=224, root=test_root, num_workers=args.num_workers)
    train_data, test_id, test_ood = data_generate()
    train_data, test_id, test_ood = data_generate()

    for alpha_value in [0.0, 0.2]:
        args.alpha = alpha_value

        # 关于T的实验
        id_acc = torch.zeros(6, 7)
        ood_acc = torch.zeros(6, 7)
        index = 0
        for num_groups, num_iterations in [[8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7]]:
            args.num_groups, args.num_iterations = num_groups, num_iterations
            for seed in range(5):
                args.seed = seed
                torch.manual_seed(seed=args.seed)
                start = time.time()
                net = resnet18(args=args)
                optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)
                result = train(net, args.batch_size, optimizer, device, args)
                id_acc[seed][index] = result[2]
                ood_acc[seed][index] = result[3]
                print('Running Time:', time.time() - start, '\n')
            index=index+1
        id_acc[5] = id_acc.sum(dim=0) / 5
        ood_acc[5] = ood_acc.sum(dim=0) / 5
        print('id_acc:', id_acc)
        print('od_acc:', ood_acc)

        # 关于G的实验
        id_acc = torch.zeros(6, 7)
        ood_acc = torch.zeros(6, 7)
        index = 0
        for num_groups, num_iterations in [[1, 5], [2, 5], [4, 5], [8, 5], [16, 5], [32, 5], [64, 5]]:
            args.num_groups, args.num_iterations = num_groups, num_iterations
            for seed in range(5):
                args.seed = seed
                torch.manual_seed(seed=args.seed)
                start = time.time()
                net = resnet18(args=args)
                optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)
                result = train(net, args.batch_size, optimizer, device, args)
                id_acc[seed][index] = result[2]*100
                ood_acc[seed][index] = result[3]*100
                print('Running Time:', time.time() - start, '\n')
            index=index+1
        id_acc[5] = id_acc.sum(dim=0) / 5
        ood_acc[5] = ood_acc.sum(dim=0) / 5
        print('id_acc:', id_acc)
        print('od_acc:', ood_acc)
    print('Total Time:', time.time() - total_time)