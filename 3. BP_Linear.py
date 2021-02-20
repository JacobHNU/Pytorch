import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# create the data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# create the Tensor for grad
w = torch.Tensor([1.0])
w.requires_grad = True  # caculate the gradient


# define the linear model
def forward(x):
    return x * w  # x*w also is a Tensor


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())

# training
epoch_list = []
loss_list = []
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)          # forward, compute the loss
        l.backward()            # the loss also is a Tensor, backward, compute grad for Tensor whose requires_grad set to True
        print('\t grad:', x, y, w.grad.item())
        w.data = w.data - 0.05 * w.grad.data      # the grad is utilized to update weight

        w.grad.data.zero_()     # the grad computed by .backward() will be accumulated,
        # so after update, remember set the grad to zero
        # we need each epoch grad respectively
        epoch_list.append(epoch)
        loss_list.append(l.item())

    print("progress:", epoch, l.item())

print("predict (after training)", 4, forward(4).item())

plt.plot(epoch_list, loss_list)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

