# -*- coding: utf-8 -*-


class Sentence:
    def __init__(self):
        self.tokens = [] # 词集合
        self.chars = 0

    def add_token(self, t):
        self.chars += len(t)
        self.tokens.append(t)

    def clear(self):
        self.tokens = []
        self.chars = 0

    # label -1, unknown
    # 0-> 'S'
    # 1-> 'B'
    # 2-> 'M'
    # 3-> 'E'
    def generate_tr_line(self, x, y):
        for t in self.tokens:
            if len(t) == 1:
                x.append(str(t[0]))
                y.append('S') # 'S'，表示单字为词
            else:
                nn = len(t)
                for i in range(nn):
                    x.append(str(t[i]))
                    if i == 0:
                        y.append('B') # 'B', 表示词的首字
                    elif i == (nn - 1):
                        y.append('E') # 'E'，表示词的尾字
                    else:
                        y.append('M') # 'M'，表示词的中间
