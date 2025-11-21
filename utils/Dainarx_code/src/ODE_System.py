import re
import numpy as np
import matplotlib.pyplot as plt
from math import *


class ODESystem:
    @staticmethod
    def analyticalExpression(expr_list: str, var_list: list[str], input_list: list[str] = None):
        if input_list is None or input_list == []:
            input_expr = ""
        else:
            input_expr = ',' + ','.join(input_list)
        pattern = r'(' + '|'.join(map(re.escape, var_list)) + r')\[(\d+)\]'
        var_pattern = r'(\S+)\[(\d+)\]'

        def repl(match):
            string = match.group(1)
            number = match.group(2)
            idx = var_list.index(string)
            return f"x[{idx}][{number}]"

        def get_info(string):
            match = re.search(var_pattern, string)
            return var_list.index(match.group(1)), int(match.group(2))

        res = [(lambda x: 0) for _ in var_list]
        order_list = [0 for _ in var_list]
        expr_list = expr_list.split(',')
        for expr in expr_list:
            var_expr, eq_expr = expr.split('=')
            eq_expr = re.sub(pattern, repl, eq_expr)
            idx, order = get_info(var_expr)
            res[idx] = eval(f"lambda x{input_expr}: " + eq_expr)
            order_list[idx] = order
        return order_list, res

    def analyticalInput(self, input_expr=None):
        res = []
        if input_expr is None:
            input_expr = {}
        for var in self.input_list:
            expr = input_expr.get(var)
            if expr is None:
                res.append(lambda t: 0)
            elif isinstance(expr, str):
                # String expression - use eval as before
                res.append(eval("lambda t: " + expr))
            elif isinstance(expr, (np.ndarray, list)):
                # Numpy array or list - create indexing function
                input_array = np.asarray(expr)
                if input_array.ndim == 1:
                    # 1D array: direct indexing
                    # Note: array[i] corresponds to time (i+1)*dt in the original data
                    # because it was collected AFTER sys.next() advanced time
                    def make_input_func(arr, dt):
                        def input_func(t):
                            if dt is None or dt <= 0:
                                return arr[0]
                            # Array index i corresponds to time (i+1)*dt
                            # So for time t, we need index i where (i+1)*dt ≈ t
                            # Thus i ≈ t/dt - 1
                            idx = int(round(t / dt - 1))
                            idx = max(0, min(idx, len(arr) - 1))  # Clamp to array bounds
                            return arr[idx]
                        return input_func
                    res.append(make_input_func(input_array, self.dt))
                elif input_array.ndim == 2:
                    # 2D array: assume shape (1, num_steps) or (num_steps, 1)
                    if input_array.shape[0] == 1:
                        arr_1d = input_array[0, :]
                    else:
                        arr_1d = input_array[:, 0]
                    # Note: array[i] corresponds to time (i+1)*dt in the original data
                    def make_input_func(arr, dt):
                        def input_func(t):
                            if dt is None or dt <= 0:
                                return arr[0]
                            # Array index i corresponds to time (i+1)*dt
                            idx = int(round(t / dt - 1))
                            idx = max(0, min(idx, len(arr) - 1))  # Clamp to array bounds
                            return arr[idx]
                        return input_func
                    res.append(make_input_func(arr_1d, self.dt))
                else:
                    raise ValueError(f"Input array for '{var}' must be 1D or 2D, got shape {input_array.shape}")
            else:
                raise TypeError(f"Input for '{var}' must be string expression, numpy array, or list, got {type(expr)}")
        return res

    def transExpr(self, expr):
        order_list, eq_list = ODESystem.analyticalExpression(self.var_list[0] + '[0] =' + expr, self.var_list)
        return eq_list[0]


    def getInput(self, t=None):
        if t is None:
            t = self.now_time
        res = [fun(t) for fun in self.input_fun_list]
        return res

    def __init__(self, expr_list, var_list, input_list=None, init_state=None, method='rk'):
        if init_state is None:
            init_state = []
        if input_list is None:
            input_list = []
        order_list, eq_list = ODESystem.analyticalExpression(expr_list, var_list, input_list)
        self.var_list = var_list
        self.input_list = input_list
        self.order_list = order_list
        self.eq_list = eq_list
        self.var_num = len(eq_list)
        self.state = init_state.copy()
        self.init_state = init_state.copy()
        self.max_order = max(self.order_list)
        self.method = method
        self.fit_state_size()
        self.input_fun_list = self.analyticalInput()
        self.now_time = 0.
        self.reset_val = {}
        self.dt = None  # Store time step for array-based inputs

    def fit_state_size(self):
        state = np.zeros((self.var_num, self.max_order)).astype(np.float64)
        for idx_var in range(min(self.var_num, len(self.state))):
            for idx_order in range(min(self.order_list[idx_var], len(self.state[idx_var]))):
                state[idx_var][idx_order] = self.state[idx_var][idx_order]
        self.state = state

    def rk_fun(self, y, t):
        res = np.roll(y, -1, axis=1)
        for idx in range(self.var_num):
            res[idx][self.order_list[idx] - 1] = self.eq_list[idx](y, *self.getInput(t))
        return res

    def next(self, dT: float):
        if self.method == "rk":
            k1 = self.rk_fun(self.state, self.now_time)
            k2 = self.rk_fun(self.state + 0.5 * dT * k1, self.now_time + 0.5 * dT)
            k3 = self.rk_fun(self.state + 0.5 * dT * k2, self.now_time + 0.5 * dT)
            k4 = self.rk_fun(self.state + dT * k3, self.now_time + dT)
            self.state = self.state + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            for idx_var in range(self.var_num):
                last_d = self.eq_list[idx_var](self.state)
                for i in range(self.order_list[idx_var] - 1, -1, -1):
                    self.state[idx_var][i] = self.state[idx_var][i] + dT * last_d
                    last_d = self.state[idx_var][i]
        self.now_time += dT
        return self.state[:, 0]

    def reset(self, init_state, dt=None):
        if dt is not None:
            self.dt = dt
        res = []
        for var in self.var_list:
            res.append(init_state.get(var, []))
        self.clear(res)
        self.input_fun_list = self.analyticalInput(init_state)

    def clear(self, init_state=None):
        if init_state is None:
            self.state = self.init_state.copy()
        else:
            self.state = init_state.copy()
        self.fit_state_size()
        self.now_time = 0.

    def load(self, other_sys, reset_dict):
        self.state = other_sys.state.copy()
        self.input_fun_list = other_sys.input_fun_list.copy()
        self.now_time = other_sys.now_time
        self.fit_state_size()
        prefix = ','.join(self.var_list)
        for i in range(len(self.var_list)):
            var = self.var_list[i]
            reset_val = reset_dict.get(var, [])
            for j in range(min(len(reset_val), self.max_order)):
                val = reset_val[j]
                if val == "":
                    continue
                elif type(val) == str:
                    self.state[i][j] = self.transExpr(val)(self.state)
                else:
                    self.state[i][j] = float(reset_val[j])


if __name__ == "__main__":
    ode = ODESystem("x1[1] = u1", ["x1"], ["u1"])
    ode.reset({"x1": [1], "u1": "1"})
    res = []
    for i in range(100):
        val = ode.next(0.01)
        res.append(val[0])
    plt.plot(np.arange(len(res)), res)
    plt.show()