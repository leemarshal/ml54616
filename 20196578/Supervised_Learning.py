import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim

import os
from PIL import Image
import argparse
from tqdm import tqdm
import time
import easydict


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
        # Output: Probability of 50 class label
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
        model = models.resnet18()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.layer4 = Identity()
        model.fc = nn.Linear(256, 50)
    elif selection == "vgg":
        model = models.vgg11_bn()
        model.features = nn.Sequential(*list(model.features.children())[:-7])
        model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=50, bias=True))
    elif selection == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=50, bias=True))
    elif selection == 'custom':
        model = Custom_model()

    return model


def train(net1, labeled_loader, optimizer, criterion, epoch):
    net1.train()
    # Supervised_training
    with tqdm(labeled_loader, desc=f'Epoch {e + 1}/{epoch}', unit='batch') as tepoch:
        start_time = time.time()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net1(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            tepoch.set_postfix({'Loss': train_loss / (batch_idx + 1), 'Acc': 100. * correct / total})

        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch Time: {:.2f} seconds".format(epoch_time))
        ####################
        # Write your Code
        # Model should be optimized based on given "targets"
        ####################


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


def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

    if not os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning')):
        os.makedirs(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning'))

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
        dataset = CustomDataset(root='./data/Supervised_Learning/labeled', transform=train_transform)
        labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        dataset = CustomDataset(root='./data/Supervised_Learning/val', transform=test_transform)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    model_name = 'mobilenet'  # Input model name to use in the model_section class
    # e.g., 'resnet', 'vgg', 'mobilenet', 'custom'

    if torch.cuda.is_available():
        model = model_selection(model_name).cuda()
    else:
        model = model_selection(model_name)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    # You may want to write a loader code that loads the model state to continue the learning process
    # Since this learning process may take a while.

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001
    step_size = 10
    gamma = 0.9
    momentum = 0.9
    epoch = 50  # Input the number of Epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # Your optimizer here
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                gamma=gamma)  # You may want to add a scheduler for your loss

    best_result = 0
    if args.test == 'False':
        assert params < 7.0, "Exceed the limit on the number of model parameters"
        start_epoch = 0  # 시작 에폭 초기화
        best_result = 0  # 최고 성능 초기화

        if os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning', 'best_model.pt')):
            # 저장된 모델 파일이 존재하는 경우
            # model.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning', 'best_model.pt'), map_location=torch.device('cuda')))
            start_epoch = load_checkpoint(model, optimizer,
                                          os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning',
                                                       'checkpoint.pt'))
            best_result = test(model, val_loader)  # 이전 최고 성능 다시 확인
            print("Loaded model from checkpoint. Starting from epoch:", start_epoch)
            print("Best result from previous training:", best_result)

        for e in range(0, epoch):
            train(model, labeled_loader, optimizer, criterion, epoch)
            tmp_res = test(model, val_loader)
            scheduler.step()
            # You can change the saving strategy, but you can't change the file name/path
            # If there's any difference to the file name/path, it will not be evaluated.
            print('{}th performance, res : {}'.format(e, tmp_res))
            if best_result < tmp_res:
                best_result = tmp_res
                torch.save(model.state_dict(),
                           os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning', 'best_model.pt'))
                save_checkpoint(model, optimizer, e,
                                os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning', 'checkpoint.pt'))
        print('Final performance {} - {}', best_result, params)

    else:
        # This part is used to evaluate.
        # Do not edit this part!
        dataset = CustomDataset(root='/data/23_1_ML_challenge/Supervised_Learning/test', transform=test_transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model.load_state_dict(
            torch.load(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning', 'best_model.pt'),
                       map_location=torch.device('cuda')))
        res = test(model, test_loader)
        print(res, ' - ', params)