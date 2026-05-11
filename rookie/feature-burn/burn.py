import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tud


def plot():
    full_data = json.load(open('./data.json', 'r'))
    means = [318.214, 3.9422493, 5.495441]
    stds = [465.4312, 3.0833914, 4.248455]
    maxs = [4579.5, 19., 28.]
    mins = [4., 4., 1.]
    xo, yo, zo, fo = [], [], [], []
    xm, ym, zm = [], [], []
    xs, ys, zs = [], [], []
    data_matrix = []
    for i, data in enumerate(full_data):
        x, y, z, f = (float(data['harvest']), float(data['engineer_no']),
                      float(data['during_fbs']), float(data['working_days']))
        xo.append(x)
        yo.append(y)
        zo.append(z)
        fo.append(f)
        xs.append((x-means[0])/stds[0])
        ys.append((y-means[1])/stds[2])
        zs.append((z-means[2])/stds[2])
        xm.append((x-mins[0])/(maxs[0]-mins[0]))
        ym.append((y-mins[1])/(maxs[1]-mins[1]))
        zm.append((z-mins[2])/(maxs[2]-mins[2]))
        data_matrix.append([x, y, z, f])

    print(np.corrcoef([np.array(xo), np.array(yo), np.array(zo), np.array(fo)]))

    ax = plt.axes()
    # ax = plt.axes(projection='3d')
    if ax.name == '3d':
        sc = ax.scatter(xo, yo, zo, s=fo, c=fo, cmap='rainbow')
    else:
        sc = ax.scatter(yo, fo, c=fo, cmap='rainbow')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if ax.name == '3d':
        ax.set_zlabel('z')

    plt.legend(*sc.legend_elements())
    plt.colorbar(sc)

    plt.show()


class FeatureDataSet(tud.Dataset):
    def __init__(self, train):
        full_data = json.load(open('./data.json', 'r'))
        if train:
            raw_data_list = full_data[0:-50]
        else:
            raw_data_list = full_data[-50:]

        self.input_feature_matrix, self.target_list = self.normalize(raw_data_list)

    @staticmethod
    def normalize(raw_data_list):
        means = [318.214, 3.9422493, 5.495441]
        stds = [465.4312, 3.0833914, 4.248455]
        maxs = [4579.5, 19., 28.]
        mins = [4., 4., 1.]

        inputs = [[float(item['harvest']), float(item['engineer_no']), float(item['during_fbs'])]
                  for item in raw_data_list]
        for i in range(3):
            # mean = numpy.mean([item[i] for item in inputs], dtype='float32')
            # std = numpy.std([item[i] for item in inputs], dtype='float32')
            # maxa = numpy.max([item[i] for item in inputs])
            # mina = numpy.min([item[i] for item in inputs])
            for data in inputs:
                data[i] = float((data[i] - mins[i]) / (maxs[i] - mins[i]))

        input_feature_matrix = []
        target_list = []
        for i, item in enumerate(inputs):
            x1, x2, x3 = item
            input_feature_matrix.append([
                x1, x1**2, x1**3, x1**4,
                x2, x2**2, x2**3, x2**4,
                x1*x2, x1*x1*x2*x2
                # x3, x3**2, x3**3, x3**4
            ])
            target_list.append([float(raw_data_list[i]['working_days'])])

        return input_feature_matrix, target_list

    def __getitem__(self, index):
        return torch.tensor(self.input_feature_matrix[index]), torch.tensor(self.target_list[index])

    def __len__(self):
        return len(self.target_list)


class FeatureNetwork(nn.Module):

    def __init__(self):
        super(FeatureNetwork, self).__init__()
        i = 1
        setattr(self, f'fc{i}', nn.Linear(10, 64))
        i += 1
        setattr(self, f'fc{i}', nn.Linear(64, 256))
        i += 1
        setattr(self, f'fc{i}', nn.Linear(256, 256))
        i += 1
        setattr(self, f'fc{i}', nn.Linear(256, 512))
        i += 1
        setattr(self, f'fc{i}', nn.Linear(512, 256))
        i += 1
        setattr(self, f'fc{i}', nn.Linear(256, 256))
        i += 1
        setattr(self, f'fc{i}', nn.Linear(256, 256))
        i += 1
        setattr(self, f'fc{i}', nn.Linear(256, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        i = 1
        x = F.relu(nn.BatchNorm1d(64, track_running_stats=True)(getattr(self, f'fc{i}')(x)))
        i += 1
        x = F.relu(nn.BatchNorm1d(256, track_running_stats=True)(getattr(self, f'fc{i}')(x)))
        i += 1
        x = F.relu(nn.BatchNorm1d(256, track_running_stats=True)(getattr(self, f'fc{i}')(x)))
        i += 1
        x = F.relu(nn.BatchNorm1d(512, track_running_stats=True)(getattr(self, f'fc{i}')(x)))
        i += 1
        x = F.relu(nn.BatchNorm1d(256, track_running_stats=True)(getattr(self, f'fc{i}')(x)))
        i += 1
        x = F.relu(nn.BatchNorm1d(256, track_running_stats=True)(getattr(self, f'fc{i}')(x)))
        i += 1
        x = F.relu(nn.BatchNorm1d(256, track_running_stats=True)(getattr(self, f'fc{i}')(x)))
        i += 1
        x = getattr(self, f'fc{i}')(x)
        return x


class FeatureTrainer:

    def __init__(self):
        self.model_path = './model.pt'
        self.net = FeatureNetwork()
        model_exist = os.path.isfile(self.model_path)
        if model_exist:
            self.net.load_state_dict(torch.load(self.model_path))
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        if not model_exist:
            self.train(tud.DataLoader(FeatureDataSet(True), 5, True, drop_last=True))

    def train(self, train_data, epochs=800):
        self.net.train()
        data_length = len(train_data)
        start = time.time()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_data, 0):
                inputs, targets = data

                self.optimizer.zero_grad()
                output = self.net(inputs)
                loss = self.criterion(output, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f'Loss in epoch {epoch + 1}: {running_loss / data_length}')
        print('Finished Training! Total cost time: ', time.time() - start)
        torch.save(self.net.state_dict(), self.model_path)

    def test(self, inputs):
        o = self.net(inputs)
        return o


if __name__ == '__main__':
    plot()
    # trainer = FeatureTrainer()
    # for i, data in enumerate(tud.DataLoader(FeatureDataSet(True), 5), 0):
    #     if i != 4:
    #         continue
    #     input, target = data
    #     o = trainer.test(input)
    #     print(o)
    #     print(target)
    #     print(nn.MSELoss()(o, target))
    #     break
