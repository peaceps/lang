import random
from copy import deepcopy

from reversi.sdk import Board
from reversi.reinforcement.trainer import ReversiTrainers
from reversi.reinforcement.utils import get_board_state, get_opposite_player, action_num


class GameSampler:

    DEFAULT_VALUED = 0.

    def __init__(self, sample_config, continue_training=False):
        self.trainers = ReversiTrainers(continue_training)
        self.rewards = sample_config['position_rewards']
        self.gama = sample_config['gama']
        self.epsilon = sample_config['max_epsilon']
        self.winner_bonus = sum(map(lambda r: sum(r), self.rewards))

    def start_sampling(self, sampling_round):
        self.trainers.start_training()
        for i in range(sampling_round):
            self.epsilon *= 1 - i / sampling_round
            self._sample_one_game()
        self.trainers.stop_training()

    def _sample_one_game(self):
        self.trainers.begin_one_game()
        board = Board()
        finished, color = False, 'X'
        while not finished:
            finished, color = self._move_next(board, color)
        self.trainers.finish_one_game()

    def _move_next(self, board, color):
        state = get_board_state(board)
        legal_actions = list(board.get_legal_actions(color))
        taken_action = self._get_action_by_greedy_epsilon(state, color, legal_actions)  # 根据greedy-ε选择下一步动作
        action_values = {}
        result = None
        for action in legal_actions:
            action_result = self._get_results_for_action(deepcopy(board), color, action)  # 获取当前棋盘所有可能动作的回报期望
            action_values[action] = action_result[0]
            if action == taken_action:
                result = action_result[1], action_result[2]

        self.trainers.train_by_one_move(color, state, legal_actions, action_values)

        board._move(taken_action, color)

        return result

    def _get_action_by_greedy_epsilon(self, state, color, actions):
        if len(actions) == 1:
            return actions[0]
        action_values = self.trainers.get_action_values(color, state, actions)  # 获取当前状态的所有动作回报
        best_action = self._get_best_action(actions, action_values)
        r1 = random.random()  # pi(a) = ε/n if a != a* else (1 - ε + ε/n)
        if r1 > self.epsilon:
            taken_action = best_action
        else:
            left_actions = list(filter(lambda a: a != best_action, actions))
            taken_action = left_actions[random.randint(0, len(left_actions)-1)]
        return taken_action

    def _get_best_action(self, actions, action_values):
        max_g = -float('inf')
        best_action = ''
        for action in actions:
            x, y = action_num(action)
            g = action_values[x][y]
            if g > max_g:
                max_g = g
                best_action = action
        return best_action

    def _get_results_for_action(self, board, color, action):
        board._move(action, color)
        i, j = board.board_num(action)
        r = self.rewards[i][j]  # 获取直接回报

        factor, new_color = -1, get_opposite_player(color)
        new_legal_actions = list(board.get_legal_actions(new_color))
        if len(new_legal_actions) == 0:  # 如果对方无子可走，本方继续落子
            factor, new_color = 1, color
            new_legal_actions = list(board.get_legal_actions(new_color))
            if len(new_legal_actions) == 0:  # 如果本方也无子可走，游戏结束，如果获胜+bonus，失败则-bonus
                win_color = 'X' if board.get_winner() == 0 else 'O' if board.get_winner() == 1 else '.'
                return (r + self.gama * self.winner_bonus
                        * (0 if win_color == '.' else 1 if win_color == color else -1), True, '')
            
        new_action_values = self.trainers.get_action_values(new_color, get_board_state(board), new_legal_actions)
        next_gain_expectation = self._get_next_actions_expectation(new_legal_actions, new_action_values)  # 获取落子后棋盘期望
        target = r + factor * self.gama * next_gain_expectation  # 计算回报
        return target, False, new_color

    def _get_next_actions_expectation(self, actions, action_expectations):
        best_action = self._get_best_action(actions, action_expectations)
        expectation = 0  # E=Σpi(a~s)Q(s,a)
        for action in actions:
            x, y = action_num(action)
            probability = 1 - self.epsilon + self.epsilon / len(actions) if action == best_action\
                else self.epsilon / len(actions)
            expectation += action_expectations[x][y] * probability
        return expectation


if __name__ == '__main__':
    config = {
        'position_rewards': [
            [100., -35., 10., 5., 5., 10., -35., 100.],
            [-35., -35., 2., 2., 2., 2., -35., -35.],
            [10., 2., 5., 1., 1., 5., 2., 10.],
            [5., 2., 1., 2., 2., 1., 2., 5.],
            [5., 2., 1., 2., 2., 1., 2., 5.],
            [10., 2., 5., 1., 1., 5., 2., 10.],
            [-35., -35., 2., 2., 2., 2., -35., -35.],
            [100., -35., 10., 5., 5., 10., -35., 100.]
        ],  # 棋盘落子回报
        'gama': 0.5,  # 行动折扣率，越大眼光越长远，越小越优先考虑近期回报
        'max_epsilon': 0.8  # 贪心率，越小越保守，优先采用当前最优策略，越大越优先探索其他路径
    }
    sampler = GameSampler(config, True)  # continue_training为True时会从当前模型开始继续训练，False则从头训练，模型及参数更改后应设为False
    sampler.start_sampling(5)
