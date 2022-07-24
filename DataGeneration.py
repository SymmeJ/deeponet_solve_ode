from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from inspect import isclass

import numpy as np
import tensorflow as tf

import deepxde as dde
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import ODESystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test

def ode_system(T,ic):
    """ODE"""

    def g(s,u,x):
        # Antiderivative
        # return u
        # Nonlinear ODE
        # return -s**2 + u
        # Gravity pendulum
        # k = 1
        # return [s[1], - k * np.sin(s[0]) + u]
        # Rober Problem
        k1,k2,k3 = 4, 3, 1
        return [(- k1 * s[0] + k3 * s[1] * s[2]) , 
                (k1 * s[0] - k2 * s[1] * s[1] - k3 * s[1] * s[2]) ,
                (k2 * s[1] * s[1]),
                u] 

    # s0 = [0]
    s0 = ic  # Gravity pendulum
    return ODESystem(g, s0, T)

def main():

    problem = "ode"
    T = 1
    
    
    space = GRF(T, length_scale=0.2, N=1000 * T, interp="cubic")

    # Hyperparameters
    m = 10000
    num_p = 100
    num_dat_once = 10
    lr = 0.001
    epochs = 50000

    # Network
    nn = "opnn"
    activation = "relu"
    initializer = "Glorot normal"  # "He normal" or "Glorot normal"
    dim_x = 1 if problem in ["ode", "lt"] else 2
    if nn == "opnn":
        net = dde.maps.OpNN(
            [m, 40, 40],
            [dim_x, 40, 40],
            activation,
            initializer,
            use_bias=True,
            stacked=False,
        )
    elif nn == "fnn":
        net = dde.maps.FNN([m + dim_x] + [100] * 2 + [1], activation, initializer)
    elif nn == "resnet":
        net = dde.maps.ResNet(m + dim_x, 1, 128, 2, activation, initializer)
### what I write
    ic1 = np.random.rand(m)
    ic2 = np.random.rand(m)
    ic3 = np.random.rand(m)
    ic4 = np.zeros(m)
    ics = np.column_stack((ic1,ic2,ic3,ic4))
    train = np.zeros(num_p+2)
    for ic in ics:
        system = ode_system(T,ic)
        X1_train, y1_train = system.gen_sensors_data(space, m, num_p,0)
        X2_train, y2_train = system.gen_xy_data(space, m, num_dat_once,1)
        X2_train = np.column_stack(X2_train)
        y1 = y1_train.reshape(1,len(y1_train))[0]
        u = np.array(list(y1)*num_dat_once).reshape(num_dat_once,num_p)
        train_once = np.column_stack((u,X2_train,y2_train))
        train = np.row_stack((train,train_once))


    
    np.savetxt('data12.dat', train)


if __name__ == "__main__":
    main()