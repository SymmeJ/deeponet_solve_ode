from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import tensorflow as tf
from system import ODESystem
from spaces import GRF

import deepxde as dde
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test

lr = 0.001
epochs = 50000
problem = "ode"
nn = "opnn"
activation = "relu"
initializer = "Glorot normal"  # "He normal" or "Glorot normal"
dim_x = 1 if problem in ["ode", "lt"] else 2

net = dde.maps.OpNN(
    [100, 40, 40],
    [1, 40, 40],
    activation,
    initializer,
    use_bias=True,
    stacked=False,
    )
dat = np.loadtxt('data_12.dat')
dat = dat[1:100001,:]
y = np.array([ele[-1] for ele in dat])
u = np.array([ele[0:-2] for ele in dat])
x = np.array([ele[-2] for ele in dat])
index = np.random.choice(100000,100000)
train_index = index[0:10000]
test_index = index[10000:100000]
x_train = x[train_index][:,None]
x_test = x[test_index][:,None]
y_train = y[train_index][:,None]
y_test = y[test_index][:,None]
u_train = u[train_index,:]
u_test = u[test_index,:]
x_train = [u_train,x_train]
x_test = [u_test,x_test]
#print(x_train.shape)
#np.savetxt("x_train.dat",x_train)
data = dde.data.OpDataSet(
            X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test
        )
model = dde.Model(data, net)
model.compile("adam", lr=lr, metrics=[mean_squared_error_outlier])

checker = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", save_better_only=True, period=1000
    )
losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])
print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
dde.saveplot(losshistory, train_state, issave=True, isplot=False)
model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
# model.restore(f"model/model-{train_state.best_step}.ckpt", verbose=1)
safe_test(model, data, x_test, y_test)


### TEST
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
T = 1
m = 100
sensors = np.linspace(0, T, num=m)[:, None]
space = GRF(T, length_scale=0.2, N=1000 * T, interp="cubic")
ics = np.column_stack((np.array([1,0.5,0.7,0.9]),
                        np.array([0,0.5,0.2,0.2]),
                        np.array([0,0.5,0.1,0.2]),
                        np.array([0,0,0,0])))
system = ode_system(T,ics[0])
a, y1_test = system.gen_sensors_data(space,1,100,0)
a, y2_test = system.gen_sensors_data(space,1,100,1)
X_test = [np.tile(y1_test.T,(100,1)),sensors]
y2_pred = model.predict(data.transform_inputs(X_test))
np.savetxt("testPred.dat", np.hstack((sensors, y2_test, y2_pred)))
print("L2relative error:", dde.metrics.l2_relative_error(y2_test, y2_pred))