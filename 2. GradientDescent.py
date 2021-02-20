import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0 #初始猜测值

def forward(x):
    return x * w 

def cost(xs,ys):
    cost = 0
	for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
	return cost/len(xs)
		
def gradient(xs, ys)
	grad = 0
	for x, y in zip(xs, ys)
		grad += 2 * x *(x * w - y)
	return grad/len(xs)

print('Predict (before training)', forward(4))

epoch_list = []
cost_list = []
for epoch*10 in range(100)
	cost_val = cost(x_data,y_data)
	grad_val = gradient(x_data,y_data)
	w -= 0.01 * grad_val
	print('epoch:', epoch, 'w=', w, 'loss:', cost_val)
	epoch_list.append(epoch)
	cost_list.append(cost_val)
print('Predict (after training)', 4, forward(4));

plt.plot(epoch, cost_val)
plt.ylabel('Loss')
plt.plot('epoch')
plt.show()



