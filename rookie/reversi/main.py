#!/usr/bin/env python3
from sdk import RandomPlayer
from sdk import Game
from reversi.reinforcement.player import TrainedPlayer, ManualPlayer
from reversi.reinforcement.utils import display_history

import datetime

if __name__ == '__main__': 
    player1 = TrainedPlayer('X')
    player2 = RandomPlayer('O')

    trained_player = 'player1' if isinstance(player1, TrainedPlayer) else\
        'player2' if isinstance(player2, TrainedPlayer) else None
    has_manual_player = isinstance(player1, ManualPlayer) or isinstance(player2, ManualPlayer)

    total_timeout = 1500000 if has_manual_player else 150
    move_timeout = 60000 if has_manual_player else 6
    game_round = 1 if has_manual_player else 100

    win_count = 0
    print_loss = False

    for _ in range(game_round):
        game = Game(player1, player2, total_timeout, move_timeout)
        start = datetime.datetime.now()
        result = game.run()
        end = datetime.datetime.now()
        spent = (end - start).total_seconds()
        if has_manual_player:
            game.board.display()
        elif result['winner'] == trained_player:
            win_count += 1
        elif print_loss:
            print('-----LOSE GAME-----')
            for history in result['boards']:
                display_history(history['data'])
                print('------------------')
        resultStr = result['result']
        print(f'{resultStr}, time spent = {spent} seconds')
    if not has_manual_player and trained_player is not None:
        print(f'Win rate is {win_count / game_round}')
