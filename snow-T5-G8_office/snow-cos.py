import time
import torch
from torch import nn
import torchvision
from torchvision import transforms
from resnet import resnet18
from methods.whitening import Whitening2dZCA, Whitening2dPCA, Whitening2dIterNorm


class Whitening(nn.Module):
    def __init__(self):
        super(Whitening, self).__init__()
        self.itn_cw = Whitening2dIterNorm(track_running_stats=False, axis=0, eps=0, iters=5)
        self.pca_cw = Whitening2dPCA(track_running_stats=False, axis=0, eps=0)
        self.zca_cw = Whitening2dZCA(track_running_stats=False, axis=0, eps=0)
        self.itn_bw = Whitening2dIterNorm(track_running_stats=False, axis=1, eps=0, iters=5)
        self.pca_bw = Whitening2dPCA(track_running_stats=False, axis=1, eps=0)
        self.zca_bw = Whitening2dZCA(track_running_stats=False, axis=1, eps=0)
        self.bn = nn.BatchNorm1d(num_features=512)


def std(sample):  # 构造单位向量
    gamma = torch.sum(sample ** 2, dim=1)
    gamma = torch.sqrt(gamma) + 1e-8
    return torch.t(torch.t(sample) / gamma)


def std_fc(features, weight, features_regular=None, weight_regular=None, std_regular=True):
    if features_regular == None:
        pass
    else:
        features = features_regular(features)
    if weight_regular == None:
        pass
    else:
        weight = weight_regular(weight)
    if std_regular == True:
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
        lable = torch.zeros(len(y), 30)
        for k in range(len(y)):
            lable[k][y[k]] = 1
        if 5 * i < 4 * len(train_pac):
            train_data.append({'X': X, 'y': lable})
        else:
            test_id.append({'X': X, 'y': lable})
        i += 1
    for X, y in test_s:
        lable = torch.zeros(len(y), 30)
        for k in range(len(y)):
            lable[k][y[k]] = 1
        test_ood.append({'X': X, 'y': lable})

    return train_data, test_id, test_ood


def evaluate_accuracy(test_data, net, weight, features_regular, weight_regular, device=None, alpha=0.1):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for i in range(len(test_data)):
            X = test_data[i]['X']
            y = test_data[i]['y']
            net.eval()
            features = net(X.to(device))
            weight = torch.zeros(10, 512).cuda()
            snow_bw = weight_regular(net.snow)
            for k in range(10):
                weight[k] = snow_bw[k]
            y_hat = std_fc(features, weight, features_regular, weight_regular=None, std_regular=True)
            acc_sum += ((y_hat.argmax(dim=1) % 10) == (y.cuda().argmax(dim=1) % 10)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
    return acc_sum / n


def train(net, batch_size, optimizer, device, num_epochs, features_regular, weight_regular, alpha=0.2):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for i in range(len(train_data)):
            X = train_data[i]['X'].to(device)
            y = train_data[i]['y'].to(device)
            features = net(X)
            weight = torch.zeros(30, 512).cuda()
            snow_bw = weight_regular(net.snow)
            # print(net.snow[0][0])# 检查权重是否有更新
            for k in range(30):
                weight[k] = snow_bw[k % 10] + alpha * snow_bw[k + 10]
            y_hat = std_fc(features, weight, features_regular, weight_regular=None, std_regular=True)
            l = ((1 - y_hat) ** 2 * y).sum() / batch_size
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            n += y.shape[0]
            batch_count += 1
        train_acc = evaluate_accuracy(train_data, net, weight, features_regular, weight_regular, alpha=alpha)
        test_id_acc = evaluate_accuracy(test_id, net, weight, features_regular, weight_regular, alpha=alpha)
        test_ood_acc = evaluate_accuracy(test_ood, net, weight, features_regular, weight_regular, alpha=alpha)
        print('epoch %d, loss %.4f, train acc %.3f, test_id acc %.3f, test_ood acc %.3f, time %.1f sec' % (
            epoch + 1, train_l_sum / batch_count, train_acc, test_id_acc, test_ood_acc, time.time() - start))


if __name__ == '__main__':
    total_time = time.time()
    torch.manual_seed(seed=23)
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr, num_epochs = 0.01, 100
    train_pac, test_pac = load_data(batch_size, resize=224, root='../Dataset/office_caltech_10_C30/train',
                                    num_workers=0)
    _, test_s = load_data(batch_size, resize=224, root='../Dataset/office_caltech_10_C30/test', num_workers=0)
    train_data, test_id, test_ood = data_generate()
    train_data, test_id, test_ood = data_generate()
    white = Whitening()
    features_regular = None
    weight_regular = white.zca_cw
    weight_decay = 0.005
    for seed in [2, 3, 5, 7, 11]:
        torch.manual_seed(seed=seed)
        print('seed:', seed)
        for alpha in [1.0]:
            print('features_regular:', features_regular)
            print('weight_regular:', weight_regular)
            print('alpha:', alpha)
            print('weight_decay:', weight_decay)
            start = time.time()
            net = resnet18(num_classes=10)
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            train(net, batch_size, optimizer, device, num_epochs, features_regular, weight_regular)
            print('Running Time:', time.time() - start, '\n')
    print('Total Time:', time.time() - total_time)
