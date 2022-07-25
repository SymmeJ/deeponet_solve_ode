# !/usr/bin/python
# coding: utf8


import os
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import netTest
import matplotlib.pyplot as plt
graphsPATH = '02.annGraph/graphs/'
fieldNames = ['U', 'T', 'N', 'O', 'NO', 'NO2']


# 训练epoch
TRAIN_TIMES = 300000
PATIENCE = 100
PRIDICTION_TOLERANCE = 0.001#1%TOLERANCE of the real data
# 输入输出的数据维度(x,y)
INPUT_FEATURE_DIM = 2
OUTPUT_FEATURE_DIM = len(fieldNames)
# 隐含层中神经元的个数
HIDDEN_1_DIM = 50 #6*50
HIDDEN_2_DIM = 50
HIDDEN_3_DIM = 50
HIDDEN_4_DIM = 50
HIDDEN_5_DIM = 50
HIDDEN_6_DIM = 50
#initial 5e-4
LEARNING_RATE = 0.0005
print(OUTPUT_FEATURE_DIM)



NFGM = []# list of dataframes
for iname in fieldNames:
    speciesI = pd.read_csv('01.orgData_LB/'+iname+'.txt', header=None)
    NFGM.append(speciesI)

#the following is used for debug
#for i, iFGM in enumerate(NFGM):
#    print(iFGM.at[1,0])
#    exit()

# ============================ step 1/6 导入数据 ============================
parser = argparse.ArgumentParser(description='give a filed name and learning rate, for example T')
parser.add_argument('--learningRate', type=float, default = LEARNING_RATE)
args = parser.parse_args()
print(args.learningRate)
LEARNING_RATE = args.learningRate


# Create directory
if not os.path.exists(graphsPATH):
    os.mkdir(graphsPATH)
    print("Directory " , graphsPATH ,  " Created ")
else:
    print("Directory " , graphsPATH ,  " already exists")




# 数据构造
# 这里x_data、y_data都是tensor格式，在PyTorch0.4版本以后，也能进行反向传播
# 所以不需要再转成Variable格式了
# linspace函数用于生成一系列数据
# unsqueeze函数可以将一维数据变成二维数据，在torch中只能处理二维数据
# The number of Z and PV for creating the input array by WZ
Ztarget = []
Ctarget = []
fhandZ = open('theZ_B.txt')
for line in fhandZ:
    Ztarget.append(float(line))
fhandC = open('theC_B.txt')
for line in fhandC:
    Ctarget.append(float(line))


x=Ztarget
y=Ctarget

inputs_np = np.zeros((len(x)*len(y),INPUT_FEATURE_DIM ))
outputs_np = np.zeros((len(x)*len(y),OUTPUT_FEATURE_DIM ))
index = -1

#数据标签的预处理
for Z in range(len(x)):
    for C in range(len(y)):
        index += 1
        inputs_np[index] = np.array([x[Z],y[C]])
        for i, iFGM in enumerate(NFGM):
            outputs_np[index][i] = iFGM.at[Z,C]#这是建立了个二维表，x和y轴分别为C和Z

#64就把.float换成.double
x_data = torch.from_numpy(inputs_np).float()
y_data = torch.from_numpy(outputs_np).float()


#下面是对标签的归一化处理
y_data_real = y_data
Min = y_data_real.min(0).values
Max = y_data_real.max(0).values
#获得数据进行还原
# print(Min)
# print(Max)
# exit()
y_data = (y_data_real-Min)/(Max-Min)

# ============================ step 2/6 选择模型 ============================

# 建立网络，初次训练
net = netTest.Batch_Net_6(INPUT_FEATURE_DIM, HIDDEN_1_DIM, HIDDEN_2_DIM, HIDDEN_3_DIM, HIDDEN_4_DIM, HIDDEN_5_DIM, HIDDEN_6_DIM, OUTPUT_FEATURE_DIM)
#加载网络，在已有的基础上继续,若没有注释掉下面的
# net.load_state_dict(torch.load('NN_PMM.pkl'))
print(net)
Stored_net = net

# ============================ step 3/6 选择优化器   =========================

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

# ============================ step 4/6 选择损失函数 =========================
# 自定义一个损失函数所需部分
# Mean Square Error (MSE), MSELoss(), L2 loss
# Mean Absolute Error (MAE), MAELoss(), L1 Loss
# The data loss
w_data = np.array(1) # the weight of data
w_data = torch.from_numpy(w_data).float()
loss_data = torch.nn.MSELoss()
# The operation loss
w_op = np.array(1)
w_op = torch.from_numpy(w_op).float()
loss_op = torch.nn.MSELoss()
# The mass conservation
total_den = 10
Const = np.zeros((len(x_data), 1))
# 创建Const矩阵，每个元素都是常数
for i in range(0, len(x_data)):
    Const[i] = total_den
Const = torch.from_numpy(Const).float()
w_G = np.array(1)
w_G = torch.from_numpy(w_G).float()
loss_G = torch.nn.MSELoss()
# The loss function should be the sum of the above loss. However, the input for them are different.
# loss_total = w_data * loss_data(NN, y_data) +
# w_op * loss_op(NN_UT, DON_UT) + w_op * loss_op(NN_p, DON_p) + w_G * loss_G(sum_NNp, total_den)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=PATIENCE)
# ============================ step 5/6 模型训练 ============================
LOSS_min = 0
for i in range(TRAIN_TIMES):
    # 输入数据进行预测
    prediction = net(x_data)
    # 损失函数处理
    NN_UT = netTest.NN_UT_help(prediction)
    NN_p = netTest.NN_p_help(prediction, OUTPUT_FEATURE_DIM - 2)
    NN_p_sum = netTest.NN_p_sum_help(NN_p)
    # 计算各个part的loss
    loss_v_data = w_data * loss_data(prediction, y_data)
    loss_v_opUT = w_op * loss_op(NN_UT, DON_UT_r) #DON_UT_r should be the value from DON
    loss_v_opP = w_op * loss_op(NN_p, DON_p_r) #DON_UT_r should be the value from DON
    loss_v_G = w_G * loss_G(NN_p_sum, Const)
    # prediction_real = prediction*(Max-Min)+Min
    # 计算预测值与真值误差，注意参数顺序问题
    # 第一个参数为预测值，第二个为真值
    loss = loss_v_data + loss_v_opUT + loss_v_opP + loss_v_G#*100/firstLoss
    # Change the learning rate
    scheduler.step(loss)
    # 开始优化步骤
    # 每次开始优化前将梯度置为0
    optimizer.zero_grad()
    # 误差反向传播
    loss.backward()
    optimizer.step()
    # 按照最小loss优化参数
    # 可视化训练结果
    # 需要算一下accuracy了
    LOSS_SQRT = np.sqrt(loss.data.numpy())
    print("Iteration : ",'%05d'%i, "\tLearningRate : {:.3e}\tLoss: {:.4e}\tRelativeError:{:.5e}".format(optimizer.param_groups[0]['lr'], loss.data.numpy(), LOSS_SQRT))
    # 获取loss最小的model
    if i == 0:
        LOSS_min = LOSS_SQRT
        Stored_net = net
    if LOSS_SQRT < LOSS_min:
        Stored_net = net
        LOSS_min = LOSS_SQRT
    # 每1000次保存一次
    if i % 1000 == 0:
        traced_script_module = torch.jit.trace(Stored_net, torch.rand(INPUT_FEATURE_DIM))
        traced_script_module.save('NN_PMM.pt')
        torch.save(Stored_net.state_dict(), 'NN_PMM.pkl')
# ============================ step 6/6 保存模型 ============================
traced_script_module = torch.jit.trace(Stored_net, torch.rand(INPUT_FEATURE_DIM))
print(LOSS_min)
traced_script_module.save('NN_PMM.pt')
torch.save(Stored_net.state_dict(), 'NN_PMM.pkl')



