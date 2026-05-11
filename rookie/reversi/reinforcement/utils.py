def get_abs_folder_path():
    return __file__.replace('\\', '/')[0:__file__.replace('\\', '/').rindex('/')]


def get_board_state(board):
    state = ''
    for row in (board['data'] if isinstance(board, dict) else board._board):
        for column in row:
            state += column
    return state


def state_num(state):
    return 0 if state == '.' else 1 if state == 'X' else 2


def action_num(action):
    return '12345678'.index(action[1].upper()), 'ABCDEFGH'.index(action[0].upper())


def num_action(x, y):
    l = [0, 1, 2, 3, 4, 5, 6, 7]
    if y in l and x in l:
        return chr(ord('A') + y) + str(x + 1)


def move_and_get_next_color(board, action, color):
    res = board._move(action, color)
    return color if res == False else get_opposite_player(color)


def get_opposite_player(color):
    return 'X' if color == 'O' else 'O'


def display_history(board_data):
    print(' ', ' '.join(list('ABCDEFGH')))
    for i in range(8):
        print(str(i + 1), ' '.join(board_data[i]))
