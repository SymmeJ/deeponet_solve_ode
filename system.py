from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pathos.pools import ProcessPool
from scipy.integrate import solve_ivp

import config

from utils import timing



     

class ODESystem(object):
    def __init__(self, g, s0, T):
        self.g = g
        self.s0 = s0
        self.T = T
    @timing
    def gen_operator_data(self, space, m, num):
        print("Generating operator data...", flush=True)
        features = space.random(num)
        sensors = np.linspace(0, self.T, num=m)[:, None]
        sensor_values = space.eval_u(features, sensors)
        x = self.T * np.random.rand(num)[:, None]
        y = self.eval_s_space(space, features, x)
        return [sensor_values, x], y
    @timing
    def gen_sensors_data(self, space, m, num,i):
        print("Generating sensors data...", flush=True)
        features = space.random(num)
        #sensors = np.linspace(0, self.T, num=m)[:, None]
        #sensor_values = space.eval_u(features, sensors)
        x = np.linspace(0,self.T,num = num)[:, None]
        y = self.eval_s_space(space, features, x,i)
        return x[None,:], y
    @timing
    def gen_xy_data(self, space, m, num,i):
        print("Generating xy data...", flush=True)
        features = space.random(num)
        #sensors = np.linspace(0, self.T, num=m)[:, None]
        #sensor_values = space.eval_u(features, sensors)
        x = self.T * np.random.rand(num)[:, None]
        y = self.eval_s_space(space, features, x,i)
        return x[None,:], y

    def eval_s_space(self, space, features, x,i):
        """For a list of functions in `space` represented by `features`
        and a list `x`, compute the corresponding list of outputs.
        """

        def f(feature, xi):
            return self.eval_s(lambda t: space.eval_u_one(feature, t), xi[0],i)

        p = ProcessPool(nodes=config.processes)
        res = p.map(f, features, x)
        return np.array(list(res))

    def eval_s_func(self, u, x,i):
        """For an input function `u` and a list `x`, compute the corresponding list of outputs.
        """
        res = map(lambda xi: self.eval_s(u, xi[0],i), x)
        return np.array(list(res))

    def eval_s(self, u, tf, i):
        """Compute `s`(`tf`) for an input function `u`.
        """

        def f(t, y):
            return self.g(y, u(t), t)

        sol = solve_ivp(f, [0, tf], self.s0, method="RK45")
        return sol.y[i, -1:]


