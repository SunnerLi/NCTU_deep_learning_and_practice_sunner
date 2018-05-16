# Simple 2048 Puzzle Game (Demo)
# Basic framework for developing 2048 programs in Python
# Author: Hung Guei (moporgic)
# Date: May 12, 2018

import random

class board:
    """simple implementation of 2048 puzzle"""
    
    def __init__(self, tile = None):
        self.tile = tile if tile is not None else [0] * 16
    
    def __str__(self):
        state = '+' + '-' * 24 + '+\n'
        for row in [self.tile[r:r + 4] for r in range(0, 16, 4)]:
            state += ('|' + ''.join('{0:6d}'.format((1 << t) & -2) for t in row) + '|\n')
        state += '+' + '-' * 24 + '+'
        return state
    
    def mirror(self):
        return board([self.tile[r + i] for r in range(0, 16, 4) for i in reversed(range(4))])
    
    def transpose(self):
        return board([self.tile[r + i] for i in range(4) for r in range(0, 16, 4)])
    
    def left(self):
        move, score = board([]), 0
        for row in [self.tile[r:r + 4] for r in range(0, 16, 4)]:
            buf = sorted(row, key = lambda t: not t) + [0]
            while buf[0]:
                if buf[0] == buf[1]:
                    buf = buf[1:] + [0]
                    buf[0] += 1
                    score += 1 << buf[0]
                move.tile += [buf[0]]
                buf = buf[1:]
            move.tile += buf[1:]
        return move, score if move.tile != self.tile else -1
    
    def right(self):
        move, score = self.mirror().left()
        return move.mirror(), score
    
    def up(self):
        move, score = self.transpose().left()
        return move.transpose(), score
    
    def down(self):
        move, score = self.transpose().right()
        return move.transpose(), score
    
    def popup(self):
        tile = self.tile[:]
        empty = [i for i, t in enumerate(tile) if not t]
        tile[random.choice(empty)] = random.choice([1] * 9 + [2])
        return board(tile)
    
if __name__ == '__main__':
    print('2048 Demo\n')
    
    state = board().popup().popup()
    score = 0
    step = 0
    while True:
        print('#{} [{}]'.format(step, score))
        print(state)
        
        moves = [state.up(), state.right(), state.down(), state.left()]
        for label, move in zip(['up', 'right', 'down', 'left'], moves):
            print('{} = {}'.format(label, move[1]))
        
        after, reward = max(moves, key = lambda move: move[1])
        if reward == -1:
            break
        state = after.popup()
        score += reward
        step += 1
        print()