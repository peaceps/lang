import os.path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

model_path = './model.pt'
data_dir = './training-data-database'


class PictureDataLoader:

    def __init__(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=False, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                       shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                      shuffle=True, num_workers=2)
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Trainer:

    def __init__(self, net, optimizer, criterion):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, train_data):
        self.net.train()
        start = time.time()
        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(train_data, 0):
                inputs, labels = data

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # 打印统计信息
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training! Total cost time: ', time.time() - start)
        torch.save(self.net.state_dict(), model_path)

    def validate(self, test_data, classes):
        self.net.eval()
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in test_data:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        print(f'Accuracy of the network on the {sum(class_total)} test images: %d %%'
              % (100 * sum(class_correct) / sum(class_total)))

    def validate_one(self, test_data, classes):
        self.net.eval()
        dataiter = iter(test_data)
        images, labels = next(dataiter)
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

        outputs = self.net(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


def imshow(img):
    img = img / 2 + 0.5  # 非归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    data_loader = PictureDataLoader()
    net = CNN()
    model_exist = os.path.isfile(model_path)
    if model_exist:
        net.load_state_dict(torch.load(model_path))
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(net, optimizer, criterion)
    if not model_exist:
        trainer.train(data_loader.trainloader)

    trainer.validate_one(data_loader.testloader, data_loader.classes)


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    main()

