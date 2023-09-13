import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models
import easydict
import os
from PIL import Image
import argparse
from tqdm import tqdm
import time


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {c: int(c) for i, c in enumerate(self.classes)}
        self.imgs = []
        for c in self.classes:
            class_dir = os.path.join(root, c)
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                self.imgs.append((path, self.class_to_idx[c]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class CustomDataset_Nolabel(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        ImageList = os.listdir(root)
        self.imgs = []
        for filename in ImageList:
            path = os.path.join(root, filename)
            self.imgs.append(path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


####################
# If you want to use your own custom model
# Write your code here
####################
class Custom_model(nn.Module):
    def __init__(self):
        super(Custom_model, self).__init__()
        # place your layers
        # CNN, MLP and etc.

    def forward(self, input):
        # place for your model
        # Input: 3* Width * Height
        # Output: Probability of 10 class label
        return predicted_label


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


####################
# Modify your code here
####################
def model_selection(selection):
    if selection == "resnet":
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.layer4 = Identity()
        model.fc = nn.Linear(256, 10)
    elif selection == "vgg":
        model = models.vgg11_bn()
        model.features = nn.Sequential(*list(model.features.children())[:-7])
        model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=10, bias=True))
    elif selection == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=10, bias=True))
    elif selection == 'custom':
        model = Custom_model()
    return model


def cotrain(net1, net2, labeled_loader, unlabeled_loader, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2,
            criterion, threshold=0.95):
    net1.train()
    net2.train()
    total_labeled = len(labeled_loader.dataset)
    total_unlabeled = len(unlabeled_loader.dataset)

    correct1 = 0
    correct2 = 0
    train_loss1 = 0
    train_loss2 = 0

    # Labeled training
    with tqdm(total=total_labeled, desc='Labeled Training', unit='batch') as pbar:
        for batch_idx, (inputs, targets) in enumerate(labeled_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            # Training for model 1
            optimizer1_1.zero_grad()
            outputs1 = net1(inputs)
            loss1 = criterion(outputs1, targets)
            loss1.backward()
            optimizer1_1.step()

            # Training for model 2
            optimizer2_1.zero_grad()
            outputs2 = net2(inputs)
            loss2 = criterion(outputs2, targets)
            loss2.backward()
            optimizer2_1.step()

            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)

            correct1 += predicted1.eq(targets).sum().item()
            correct2 += predicted2.eq(targets).sum().item()

            train_loss1 += loss1.item()
            train_loss2 += loss2.item()

            pbar.set_postfix({'M1 Loss': train_loss1 / (batch_idx + 1), 'M2 Loss': train_loss2 / (batch_idx + 1),
                              'M1 Acc': 100. * correct1 / total_labeled, 'M2 Acc': 100. * correct2 / total_labeled})
            pbar.update(inputs.shape[0])

    # Unlabeled training
    with tqdm(total=total_unlabeled, desc='Unlabeled Training', unit='batch') as pbar:
        for batch_idx, inputs in enumerate(unlabeled_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            # We don't want to compute gradients with respect to these operations
            with torch.no_grad():
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)

                # Get the probabilities using softmax
                probs1 = torch.nn.functional.softmax(outputs1, dim=1)
                probs2 = torch.nn.functional.softmax(outputs2, dim=1)

                # Get the predictions
                _, preds1 = torch.max(probs1, 1)
                _, preds2 = torch.max(probs2, 1)

                # Get the indices where the models agree on the label
                agree = preds1 == preds2

                # Get the pseudo labels where the models agree and the probability is above the threshold
                pseudo_labels = preds1[agree & ((probs1.max(1)[0] > threshold) | (probs2.max(1)[0] > threshold))]

                # Get the inputs for which we have pseudo labels
                new_inputs = inputs[agree & ((probs1.max(1)[0] > threshold) | (probs2.max(1)[0] > threshold))]

            # Only perform a training step if there are pseudo labels
            if len(pseudo_labels) > 0:
                new_inputs, pseudo_labels = new_inputs.cuda(), pseudo_labels.cuda()

                # Unlabeled training for model 1
                optimizer1_2.zero_grad()
                outputs = net1(new_inputs)
                loss = criterion(outputs, pseudo_labels)
                loss.backward()
                optimizer1_2.step()

                # Unlabeled training for model 2
                optimizer2_2.zero_grad()
                outputs = net2(new_inputs)
                loss = criterion(outputs, pseudo_labels)
                loss.backward()
                optimizer2_2.step()

            pbar.update(inputs.shape[0])


def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return 100. * correct / total


def save_checkpoint(model1, model2, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2, scheduler1_1, scheduler1_2,
                    scheduler2_1, scheduler2_2, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model1_state_dict': model1.state_dict(),
        'model2_state_dict': model2.state_dict(),
        'optimizer1_1_state_dict': optimizer1_1.state_dict(),
        'optimizer1_2_state_dict': optimizer1_2.state_dict(),
        'optimizer2_1_state_dict': optimizer2_1.state_dict(),
        'optimizer2_2_state_dict': optimizer2_2.state_dict(),
        'scheduler1_1_state_dict': scheduler1_1.state_dict(),
        'scheduler1_2_state_dict': scheduler1_2.state_dict(),
        'scheduler2_1_state_dict': scheduler2_1.state_dict(),
        'scheduler2_2_state_dict': scheduler2_2.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model1, model2, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2, scheduler1_1, scheduler1_2,
                    scheduler2_1, scheduler2_2, filename):
    checkpoint = torch.load(filename, map_location=torch.device('cuda'))
    model1.load_state_dict(checkpoint['model1_state_dict'])
    model2.load_state_dict(checkpoint['model2_state_dict'])
    optimizer1_1.load_state_dict(checkpoint['optimizer1_1_state_dict'])
    optimizer1_2.load_state_dict(checkpoint['optimizer1_2_state_dict'])
    optimizer2_1.load_state_dict(checkpoint['optimizer2_1_state_dict'])
    optimizer2_2.load_state_dict(checkpoint['optimizer2_2_state_dict'])
    scheduler1_1.load_state_dict(checkpoint['scheduler1_1_state_dict'])
    scheduler1_2.load_state_dict(checkpoint['scheduler1_2_state_dict'])
    scheduler2_1.load_state_dict(checkpoint['scheduler2_1_state_dict'])
    scheduler2_2.load_state_dict(checkpoint['scheduler2_2_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='False')
    parser.add_argument('--student_abs_path', type=str, default='./')
    args = parser.parse_args()
    '''args = easydict.EasyDict({
        "test": 'False',
        "student_abs_path": './'
    })'''

    batch_size = 32  # Input the number of batch size
    if args.test == 'False':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = CustomDataset(root='./data/Semi-Supervised_Learning/labeled', transform=train_transform)
        labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        dataset = CustomDataset_Nolabel(root='./data/Semi-Supervised_Learning/unlabeled', transform=train_transform)
        unlabeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        dataset = CustomDataset(root='./data/Semi-Supervised_Learning/val', transform=test_transform)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if not os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning')):
        os.makedirs(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning'))

    model_sel_1 = 'mobilenet'  # write your choice of model (e.g., 'vgg')
    model_sel_2 = 'resnet'  # write your choice of model (e.g., 'resnet)

    model1 = model_selection(model_sel_1)
    model2 = model_selection(model_sel_2)

    params_1 = sum(p.numel() for p in model1.parameters() if p.requires_grad) / 1e6
    params_2 = sum(p.numel() for p in model2.parameters() if p.requires_grad) / 1e6

    if torch.cuda.is_available():
        model1 = model1.cuda()
    if torch.cuda.is_available():
        model2 = model2.cuda()

    # You may want to write a loader code that loads the model state to continue the learning process
    # Since this learning process may take a while.

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer1_1 = torch.optim.SGD(model1.parameters(), lr=0.001,
                                   momentum=0.9)  # Optimizer for model 1 in labeled training
    optimizer2_1 = torch.optim.Adam(model2.parameters(), lr=0.0003)  # Optimizer for model 2 in labeled training

    optimizer1_2 = torch.optim.SGD(model1.parameters(), lr=0.001,
                                   momentum=0.9)  # Optimizer for model 1 in unlabeled training
    optimizer2_2 = torch.optim.Adam(model2.parameters(), lr=0.0003)  # Optimizer for model 2 in unlabeled training

    scheduler1_1 = torch.optim.lr_scheduler.StepLR(optimizer1_1, step_size=10, gamma=0.9)
    scheduler2_1 = torch.optim.lr_scheduler.StepLR(optimizer2_1, step_size=10, gamma=0.99)
    scheduler1_2 = torch.optim.lr_scheduler.StepLR(optimizer1_2, step_size=10, gamma=0.9)
    scheduler2_2 = torch.optim.lr_scheduler.StepLR(optimizer2_2, step_size=10, gamma=0.99)

    epoch = 400  # Input the number of epochs

    if args.test == 'False':
        assert params_1 < 7.0, "Exceed the limit on the number of model_1 parameters"
        assert params_2 < 7.0, "Exceed the limit on the number of model_2 parameters"

        start_epoch = 0  # 시작 에폭 초기화
        best_result_1 = 0
        best_result_2 = 0

        if os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'checkpoint.pt')):
            # 저장된 모델 파일이 존재하는 경우
            # model.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning', 'best_model.pt'), map_location=torch.device('cuda')))
            start_epoch = load_checkpoint(model1, model2, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2,
                                          scheduler1_1, scheduler1_2, scheduler2_1, scheduler2_2, \
                                          os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning',
                                                       'checkpoint.pt'))
            best_result_1 = test(model1, val_loader)  # 이전 최고 성능 다시 확인
            best_result_2 = test(model2, val_loader)  # 이전 최고 성능 다시 확인
            print("Loaded model from checkpoint. Starting from epoch:", start_epoch)
            print("Best result from previous training:", best_result_1, best_result_2)

        for e in range(start_epoch, epoch):
            cotrain(model1, model2, labeled_loader, unlabeled_loader, optimizer1_1, optimizer1_2, optimizer2_1,
                    optimizer2_2, criterion)
            # Update learning rate
            scheduler1_1.step()
            scheduler2_1.step()
            scheduler1_2.step()
            scheduler2_2.step()
            tmp_res_1 = test(model1, val_loader)
            # You can change the saving strategy, but you can't change file name/path for each model
            print("[{}th epoch, model_1] ACC : {}".format(e, tmp_res_1))
            if best_result_1 < tmp_res_1:
                best_result_1 = tmp_res_1
                torch.save(model1.state_dict(), os.path.join('./logs', 'Semi-Supervised_Learning', 'best_model_1.pt'))

            tmp_res_2 = test(model2, val_loader)
            # You can change save strategy, but you can't change file name/path for each model
            print("[{}th epoch, model_2] ACC : {}".format(e, tmp_res_2))
            if best_result_2 < tmp_res_2:
                best_result_2 = tmp_res_2
                torch.save(model2.state_dict(), os.path.join('./logs', 'Semi-Supervised_Learning', 'best_model_2.pt'))

            save_checkpoint(model1, model2, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2, scheduler1_1,
                            scheduler1_2, scheduler2_1, scheduler2_2, e,
                            os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'checkpoint.pt'))
        print('Final performance {} - {}  // {} - {}', best_result_1, params_1, best_result_2, params_2)


    else:
        dataset = CustomDataset(root='/data/23_1_ML_challenge/Semi-Supervised_Learning/test', transform=test_transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model1.load_state_dict(
            torch.load(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'best_model_1.pt'),
                       map_location=torch.device('cuda')))
        res1 = test(model1, test_loader)

        model2.load_state_dict(
            torch.load(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'best_model_2.pt'),
                       map_location=torch.device('cuda')))
        res2 = test(model2, test_loader)

        if res1 > res2:
            best_res = res1
            best_params = params_1
        else:
            best_res = res2
            best_params = params_2

        print(best_res, ' - ', best_params)