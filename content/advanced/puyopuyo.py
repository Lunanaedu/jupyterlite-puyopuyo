import numpy as np
import random

class CFG:
    Height = 12
    Width = 6
    n_color= 4

    rensaBonus = [
        0, 8, 16, 32, 64, 96, 128, 160, 192, 224,
        256, 288, 320, 352, 384, 416, 448, 480, 512,
        544, 576, 608, 640, 672]
    pieceBonus = [0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 10, 10]
    colorBonus = [0, 0, 3, 6, 12, 24]


class Puyopuyo:
    def __init__(self):
        self.x = 2
        self.y = 0
        self.dx = [1,  0, -1, 0]
        self.dy = [0, -1,  0, 1]
        self.centerPuyo = random.randint(1, CFG.n_color)
        self.movablePuyo = random.randint(1, CFG.n_color)
        self.rotation = 1    

class EnvPuyopuyo:

   def __init__(self, height=CFG.Height, width=CFG.Width ):
      self.height = height
      self.width = width
      self.reset()

   def reset(self):
      self.board = np.zeros(self.height * self.width, dtype = np.int32).reshape(self.height, self.width)
      self.puyo, self.done = utils.create_new_puyo(self.board)      
      return self.board, self.puyo

   def step(self, action):
      self.puyo.x = action[0]
      self.puyo.rotation = action[1]
      if utils.check_collision(self.board, self.puyo):
         self.reset()
         return self.board, self.puyo, 0, True
      
      self.board = utils.set_puyo_to_board(self.board,self.puyo)
      reward = 0
      rensa = 0
      while True:
         utils.fall(self.board)
         clear_groups = utils.check_erase(self.board)
         if not len(clear_groups):
            break
         piece, color = utils.erasing(self.board, clear_groups)
         rensa += 1
         reward += utils.calc_score(rensa, piece, color)
      self.puyo, self.done= utils.create_new_puyo(self.board)

      return self.board, self.puyo, reward, self.done


class Agents:
    def random_agent(board, puyo):
        action_list = utils.create_action_list(board)
        if len(action_list) == 0:
            return [2,1]
        random_id = random.randint(0, len(action_list)-1)
        return action_list[random_id]


class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = np.arange(n)

    def find(self, x):
        if self.parents[x] == x:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def merge(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if x > y:
            x, y = y, x
        self.parents[y] = x

    def clear_groups(self, limit):
        clear_list=[]
        uf_list = [[] for _ in range(self.n)]
        for i in range(self.n):
            pa = self.find(i)
            uf_list[pa].append(i)
        for i in range(self.n):
            if len(uf_list[i]) >= limit:
                clear_list.append(uf_list[i])
        return clear_list


class utils:
    def create_sample_board(height=CFG.Height, width=CFG.Width):
        sample_list = np.arange(width)
        random.shuffle(sample_list)
        board = np.zeros(height * width, dtype = np.int32).reshape(height, width)

        for j in range(width):
            if sample_list[j]:
                for i in range(sample_list[j]):
                    board[height - 1 - i, j] = random.randint(1, CFG.n_color)

        return board

    def create_new_puyo(board):
        new_puyo = Puyopuyo()
        done = False
        if board[2, 0] > 0:
            done = True
        return new_puyo, done    

    def set_puyo_to_board(board, puyo):
        new_board = np.copy(board)
        new_board[puyo.y, puyo.x ] = puyo.centerPuyo
        puyo_dy = puyo.y + puyo.dy[puyo.rotation]
        puyo_dx = puyo.x + puyo.dx[puyo.rotation]
        if puyo_dy >= 0:
            new_board[puyo_dy, puyo_dx ] = puyo.movablePuyo
        return new_board

    def check_collision(board, puyo):
        rot = puyo.rotation
        if rot == 0 and puyo.x == 5:
            return True
        if rot == 2 and puyo.x == 0:
            return True
        if puyo.y >= 12:
            return True
        if puyo.y == 11 and rot == 3 :
            return True
        if board[puyo.y, puyo.x] > 0 :
            return True
        if not( rot == 1) and board[puyo.y + puyo.dy[rot], puyo.x + puyo.dx[rot]] > 0:
            return True
        return False
    

    def create_action_list(board):
        puyo2 = Puyopuyo()
        res = []
        for rot in range(4):
            for pos1 in range(6):
                puyo2.x = pos1
                puyo2.rotation = rot
                if not utils.check_collision(board, puyo2):   
                    res.append([pos1, rot])
        return res

    def auto_fix_puyo(board, puyo):
        for i in range(CFG.Height):
            puyo.y = i
            if utils.check_collision(board, puyo):
                puyo.y -= 1
                break
        new_board = utils.set_puyo_to_board(board, puyo)
        return new_board
    
    def fall(board):
        for j in range(CFG.Width):
            target_row = CFG.Height - 1
            for i in range(CFG.Height-1,-1,-1):
                if board[i,j] > 0:
                    if target_row > i:
                        board[target_row,j] = board[i, j]
                        board[i, j] = 0
                    target_row -= 1

    def check_erase(board, height=CFG.Height, width=CFG.Width):
        uf = UnionFind(height * width)
        
        for j in range(width):
            for i in range(height-1, -1, -1):
                if board[i, j] == 0:
                    break

                if i > 0 and board[i, j] == board[i - 1, j]:
                    uf.merge(width * i + j, width * (i - 1) + j )
                if j < width - 1 and board[i, j]==board[i, j + 1]:
                    uf.merge(width * i + j, width * i + j + 1)
        
        return uf.clear_groups(4)


    def erasing(board, clear_groups, height=CFG.Height, width=CFG.Width):
        if len(clear_groups) == 0:
            return 0, 0
        color = np.zeros(6)
        piece = 0
        color_num = 0
        for item in clear_groups:
            x, y = item[0] % width, item[0] // width
            c1 = board[y, x]
            color[c1] = 1
            for item2 in item:
                x, y = item2 % width, item2 // width
                board[y, x] = 0
                piece +=1
        for i in range(6):
            if color[i]:
                color_num += 1
        return piece, color_num

    def calc_score(rensa, piece, color):
        rensa = min(rensa, len(CFG.rensaBonus) - 1)
        piece = min(piece, len(CFG.pieceBonus) - 1)
        color = min(color, len(CFG.colorBonus) - 1)

        scale = CFG.rensaBonus[rensa] + CFG.pieceBonus[piece] + CFG.colorBonus[color]
        if scale == 0:
            scale = 1
        return scale * piece * 10

    
    def next_board(board, puyo, action):
        puyo.x = action[0]
        puyo.rotation = action[1]
        if utils.check_collision(board, puyo):
            return board, 0, True
        new_board = utils.set_puyo_to_board(board, puyo)
    
        reward = 0
        rensa = 0
        while True:
            utils.fall(new_board)
            clear_groups = utils.check_erase(new_board)
            if not len(clear_groups):
                break
            piece, color = utils.erasing(new_board, clear_groups)
            rensa += 1
            reward += utils.calc_score(rensa, piece, color)

        return new_board, reward, False

