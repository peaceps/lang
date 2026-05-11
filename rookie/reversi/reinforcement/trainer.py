import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reversi.sdk import Board
from reversi.reinforcement.utils import (get_board_state, state_num, action_num, num_action,
                                         move_and_get_next_color, get_abs_folder_path)


folder_path = get_abs_folder_path()
model_path = {'X': f'{folder_path}/models/model_X.pt', 'O': f'{folder_path}/models/model_O.pt'}


class GameDataLoader:

    # 2*8*8 第一个矩阵为当前落子状态，第二个矩阵为合法落子位置
    @staticmethod
    def convert_state_to_tensor(state, actions):
        num_state = list(map(state_num, state))
        state_tensor = torch.tensor(num_state).reshape(8, -1)
        action_tensor = torch.zeros(state_tensor.size())

        for action in actions:
            x, y = action_num(action)
            action_tensor[x][y] += 1
        input_tensor = torch.stack([state_tensor, action_tensor])
        return input_tensor


class GameNetwork(nn.Module):

    def __init__(self):
        super(GameNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 64)

    def forward(self, x):
        actions = x[1].clone().detach()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(8, -1)  # 1*64 -> 8*8
        x = x * actions  # 将非法位置归0

        return x


class GameTrainer:

    def __init__(self, color, load_model=False, evaluate=False):
        self.color = color
        self.net = GameNetwork()
        if load_model:
            self.net.load_state_dict(torch.load(model_path[color]))
        if evaluate:
            self.net.eval()
        else:
            self.net.train()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.training_info = None

    def start_training(self):
        self.training_info = {'start_time': time.time(), 'game_count': 0}

    def begin_one_game(self):
        self.training_info['game_count'] += 1
        self.training_info['running_loss'] = 0.
        self.training_info['step_count'] = 0

    def finish_one_game(self):
        average_loss = self.training_info['running_loss'] / self.training_info['step_count']
        print('Trainer %s game [%d] loss: %.3f' % (self.color, self.training_info['game_count'], average_loss))

    def stop_training(self):
        print(f'Finished Training! Total cost time: {time.time() - self.training_info["start_time"]}')
        torch.save(self.net.state_dict(), model_path[self.color])

    def train_by_one_move(self, state, actions, action_values):
        input = GameDataLoader.convert_state_to_tensor(state, actions)
        target = torch.zeros(8, 8)
        for action in action_values:  # 计算结果张量
            x, y = action_num(action)
            target[x][y] += action_values[action]
        self.optimizer.zero_grad()
        output = self.net(input)
        loss = self.criterion(output, target)
        self.training_info['running_loss'] += loss.item()
        self.training_info['step_count'] += 1
        loss.backward()
        self.optimizer.step()

    def get_value_of_state_actions(self, state, actions):
        input = GameDataLoader.convert_state_to_tensor(state, actions)
        return self.net(input).clone().detach().numpy().tolist()

    def get_next_move(self, board):
        state = get_board_state(board)
        actions = board.get_legal_actions(self.color)
        inputs = GameDataLoader.convert_state_to_tensor(state, actions)
        outputs = self.net(inputs)
        min_value = torch.min(torch.min(outputs, 0)[0], 0)[0].item()
        outputs += inputs[1] * ((0 if min_value > 0 else -1 * min_value) + 1)  # 合法位置最小值小于0则加正
        max_0 = outputs.max(0)
        max_v = max_0[0].max(0)[1].item()
        x, y = max_0[1][max_v].item(), max_v  # 取最大值
        return num_action(x, y)


class ReversiTrainers:

    def __init__(self, load_model=False, evaluate=False):
        self.trainers = {
            'X': GameTrainer('X', load_model, evaluate),
            'O': GameTrainer('O', load_model, evaluate)
        }

    def start_training(self):
        for trainer in self.trainers.values():
            trainer.start_training()

    def stop_training(self):
        for trainer in self.trainers.values():
            trainer.stop_training()

    def begin_one_game(self):
        for trainer in self.trainers.values():
            trainer.begin_one_game()

    def finish_one_game(self):
        for trainer in self.trainers.values():
            trainer.finish_one_game()

    def get_action_values(self, color, state, actions):
        return self.trainers[color].get_value_of_state_actions(state, actions)

    def train_by_one_move(self, color, state, actions, action_values):
        self.trainers[color].train_by_one_move(state, actions, action_values)

    def get_next_move(self, color, board):
        return self.trainers[color].get_next_move(board)


def validate_moves(trainers):
    board = Board()
    color = 'X'
    color = move_and_get_next_color(board, 'F5', color)
    color = move_and_get_next_color(board, 'F6', color)
    board.display()
    next_move = trainers.get_next_move(color, board)
    print(f'next color is {color} and move is {next_move}')


if __name__ == '__main__':
    reversi_trainers = ReversiTrainers(True, True)
    validate_moves(reversi_trainers)
