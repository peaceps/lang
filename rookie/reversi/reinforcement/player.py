from reversi.sdk.player import Player
from reversi.reinforcement.trainer import GameTrainer


class TrainedPlayer(Player):

    def __init__(self, color):
        super().__init__(color)
        self.trainer = GameTrainer(color, True, True)

    def get_move(self, board):
        action = self.trainer.get_next_move(board)
        return action


class ManualPlayer(Player):

    def get_move(self, board):
        board.display()
        next_action = input('Please input your move:')
        return next_action
