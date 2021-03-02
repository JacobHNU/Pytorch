import torch
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# create the data
x_data = [1.0, 2.0, 3.0]
y_data = [4.0, 9.0, 16.0]

# create the Tensor for grad
w1 = torch.Tensor([1.0])  # 初始权值
w1.requires_grad = True
w2 = torch.Tensor([1.0])
w2.requires_grad = True
b = torch.Tensor([1.0])
b.requires_grad = True


# define the linear model
def forward(x):
    return x * x * w1 + x * w2 + b  # x^2*w1 + x * w2  also is a Tensor     w1=1, w2=2, b=1


def loss(x, y):  # 构建计算图
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())

# training
epoch_list = []
loss_list = []
for epoch in range(1000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\t grad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data  #注意这里的grad是一个tensor，所以要取他的data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_() #释放之前计算的梯度
        w2.grad.data.zero_()
        b.grad.data.zero_()
        epoch_list.append(epoch)
        loss_list.append(l.item())

    print("progress:", epoch, l.item())
print("predict (after training)", 4, forward(4).item())

plt.plot(epoch_list, loss_list)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()


#predict (before training) 4 21.0
#训练1000epoch
#predict (after training) 4 25.152122497558594
#训练10000epoch
#predict (after training) 4 25.000938415527344
#在修改训练集的x,y值后，训练结果接近25，
