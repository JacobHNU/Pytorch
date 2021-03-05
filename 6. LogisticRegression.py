import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 第一步，准备数据集
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# 第二步，封装模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))  #此处用torch
        return y_pred


model = LogisticRegressionModel()

# 第三步，构造损失函数和优化器
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 第四步，训练
for epoch in range(1000):
    y_pred = model(x_data)  # 前向传播
    loss = criterion(y_pred, y_data)  # 计算loss
    print(epoch, loss.item())

    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

x = np.linspace(0, 10, 200)  #从0开始，到10结束，共返回200个数
x_t = torch.Tensor(x).view((200, 1))  # 维度为200*1，view()的作用相当于numpy中的reshape，重新定义矩阵的形状。因为此处还是linearmodel所以就改变输入维度为200*1
y_t = model(x_t)            #测试
y = y_t.data.numpy()    #numpy格式的数据
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')   # [0,10] 表示x轴，c = 'r'表明为红色的线
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()

