import re

import numpy as np
from src.DE import DE
from math import *


class FeatureExtractor:
    @staticmethod
    def unfoldItem(expr_list, idx, var_num):
        res = []
        expr_list = expr_list.copy()
        for expr in expr_list:
            expr = re.sub(r'x\[', 'x[' + str(idx) + '][', expr)
            expr = re.sub(r'x(\d)', r'x[\1]', expr)
            if 'x_' not in expr:
                res.append(expr)
                continue
            for i in range(var_num):
                if i == idx:
                    continue
                res.append(re.sub(r'x_\[', 'x[' + str(i) + '][', expr))
        return res

    @staticmethod
    def unfoldDigit(expr_list, order):
        res = []
        for expr in expr_list:
            for i in range(order):
                res.append(re.sub(r'\[\?', r"[" + str(i), expr))
        return res

    @staticmethod
    def extractValidExpression(expr_list, idx):
        res = []
        for expr in expr_list:
            if expr.isspace() or expr == '':
                continue
            expr = re.sub(r'\[(\d+)', lambda x: '[' + str(int(x.group(1)) - 1), expr)
            if ':' not in expr:
                res.append(expr)
            else:
                filed, s = expr.split(':')
                if 'x' + str(idx) in filed:
                    res.append(s)
        return res

    @staticmethod
    def findMaxorder(expr):
        numbers = re.findall(r']\[(\d+)', expr)
        return max([int(num) for num in numbers])

    @staticmethod
    def analyticalExpression(expr_list: str, var_num, order):
        res = [[] for _ in range(var_num)]
        res_order = [[] for _ in range(var_num)]
        expr_list = expr_list.split(';')
        for idx in range(var_num):
            expr = FeatureExtractor.extractValidExpression(expr_list, idx)
            expr = FeatureExtractor.unfoldDigit(expr, order)
            expr = FeatureExtractor.unfoldItem(expr, idx, var_num)
            for s in expr:
                res[idx].append(eval('lambda x: ' + s))
                res_order[idx].append(FeatureExtractor.findMaxorder(s) + 1)
        return res, res_order

    def __init__(self, var_num: int, input_num: int, order: int, dt: float,
                 need_bias: bool = False, minus: bool = False, other_items: str = ''):
        self.var_num = var_num
        self.order = order
        self.dt = dt
        self.input_num = input_num
        self.minus = minus
        self.need_bias = need_bias
        self.fun_list, self.fun_order = FeatureExtractor.analyticalExpression(other_items, var_num, order)

    def get_eps(self, data):
        return 1e-6 * self.dt * np.max(data)

    def get_items(self, data, input_data, idx, max_order=None):
        res = []
        if max_order is None:
            max_order = self.order
        max_order = min(self.order, max_order)
        for i in range(len(data[idx]) - self.order, len(data[idx]) - self.order + max_order):
            res.append(data[idx][i])
        for i in range(len(data[idx]) - self.order + max_order, len(data[idx])):
            res.append(0.)
        for (fun, order) in zip(self.fun_list[idx], self.fun_order[idx]):
            if order > max_order:
                res.append(0.)
            else:
                res.append(fun(data))
        for input_idx in range(len(input_data)):
            res.append(input_data[input_idx][0])
        if self.need_bias:
            res.append(1.)
        return res

    def work_minus(self, data, input_data, is_list: bool):
        res = []
        err = []
        max_order = []
        for idx in range(self.var_num):
            now_order = 1
            while True:
                a, b = [], []
                if is_list:
                    for block, block_input in zip(data, input_data):
                        self.append_data_only(a, b, block, block_input, idx, now_order)
                else:
                    self.append_data_only(a, b, data, input_data, idx, now_order)
                x = np.linalg.lstsq(a, b, rcond=None)[0]
                a, b = np.array(a), np.array(b)
                now_err = max(np.abs((a @ x) - b))
                if now_order == self.order or now_err < 1e-8:
                    res.append(x)
                    max_order.append(now_order)
                    err.append(now_err)
                    break
                now_order += 1
        return res, err, max_order

    def work_normal(self, data, input_data, is_list: bool):
        res = []
        err = []
        var_num = len(data) if not is_list else len(data[0])
        matrix_list = [[] for _ in range(var_num)]
        b_list = [[] for _ in range(var_num)]
        if is_list:
            for block, block_input in zip(data, input_data):
                self.append_data(matrix_list, b_list, block, block_input)
        else:
            self.append_data(matrix_list, b_list, data, input_data)
        for a, b in zip(matrix_list, b_list):
            x = np.linalg.lstsq(a, b, rcond=None)[0]
            res.append(x)
            err.append(max(np.abs((a @ x) - b)))
        return res, err, [self.order for _ in range(self.var_num)]

    def __call__(self, data, input_data, is_list=False):
        if self.minus:
            return self.work_minus(data, input_data, is_list)
        else:
            return self.work_normal(data, input_data, is_list)

    def append_data(self, matrix_list, b_list, data: np.array, input_data):
        data = np.array(data)
        input_data = np.array(input_data)
        for i in range(len(data[0]) - self.order):
            if i == 0:
                this_line = data[:, (self.order - 1)::-1]
                this_line_input = input_data[:, self.order::-1]
            else:
                this_line = data[:, (self.order + i - 1):(i - 1):-1]
                this_line_input = input_data[:, (self.order + i):(i - 1):-1]
            for idx in range(len(this_line)):
                matrix_list[idx].append(self.get_items(this_line, this_line_input, idx))
                b_list[idx].append(data[idx][i + self.order])

    def append_data_only(self, matrix_a, b, data: np.array, input_data, idx, max_order=None):
        data = np.array(data)
        for i in range(len(data[0]) - self.order):
            if i == 0:
                this_line = data[:, (self.order - 1)::-1]
                this_line_input = input_data[:, self.order::-1]
            else:
                this_line = data[:, (self.order + i - 1):(i - 1):-1]
                this_line_input = input_data[:, (self.order + i):(i - 1):-1]
            matrix_a.append(self.get_items(this_line, this_line_input, idx, max_order))
            b.append(data[idx][i + self.order])
