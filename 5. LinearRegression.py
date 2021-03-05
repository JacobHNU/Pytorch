import torch

#第一步，准备数据集
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

#第二步，封装成模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()

#第三步，构造损失函数和优化器
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#第四步，训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

#测试训练的模型的参数
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred=',y_test.data)    

#训练结果  SGD
#w= 1.999724268913269
#b= 0.0006267911521717906
#测试结果
#y_pred= tensor([[7.9995]])

# Adam
#w= 1.943166971206665
#b= 0.12614794075489044
#y_pred= tensor([[7.8988]])

# ASGD
#w= 1.9994657039642334
#b= 0.0012008678168058395
#y_pred= tensor([[7.9991]])