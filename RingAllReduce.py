import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from math import ceil
import time
import torchvision.models as models
import numpy as np
LR = 0.05
NUM_WORKER = 4
BZ = 32
EPOCH = 1
ITERATION = 1000
SHOW_LOSS = 50
SHOW_ACC = 400
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):  # 训练 ALexNet
    def __init__(self):
        super(Net, self).__init__()
        # 五个卷积层
        self.conv1 = nn.Sequential(  # 输入 32 * 32 * 3
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (32-2)/2+1 = 16
        )
        self.conv2 = nn.Sequential(  # 输入 16 * 16 * 6
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (16-2)/2+1 = 8
        )
        self.conv3 = nn.Sequential(  # 输入 8 * 8 * 16
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(  # 输入 4 * 4 * 64
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(  # 输入 2 * 2 * 128
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (2-2)/2+1 = 1
        )  # 最后一层卷积层，输出 1 * 1 * 128
        # 全连接层
        self.dense = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 4096)
        x = self.dense(x)
        return x

def show_model_detail(net):
    for i, n in enumerate(net.parameters()):
        print("for layer {0:2d}, the size of the model is {1}".format(int(i+1), n.data.size()))
    segmentation = [int((i+1)/NUM_WORKER) for j in range(NUM_WORKER)]
    for j in range(i + 1  - int((i + 1) / NUM_WORKER ) * NUM_WORKER):
        segmentation[j] += 1
    print("We have " + str(i + 1) + " layers")
    print("Seperation result: ",segmentation)
    return segmentation

class Worker(object):
    def __init__(self, segmentation, id):
        self.net = Net().to(DEVICE)
        self.id = id   #id can be zero
        self.round_of_step = len(segmentation)
        self.segmentation_result = [list(range(j)) for j in segmentation]
        for i in range(len(segmentation)-1):
            for j in range(len(self.segmentation_result[i+1])):
                self.segmentation_result[i+1][j] += self.segmentation_result[i][-1] + 1
        print("model layer segmentation result:", self.segmentation_result)
        self.criterion = torch.nn.CrossEntropyLoss()

    def model_step(self, data):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        self.net.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.aggregated_grad = []
        self.parameter = []
        for i, model in enumerate(self.net.parameters()):
            self.aggregated_grad.append(model.grad.clone())
            self.parameter.append(model.data.clone())
        return loss.item()

    def update_model_scatter_send(self, round):
        """
        The round th step of the scatter stage

        :param round: this is the round th step, round can be zero
        :return: scatter local model
        """
        with torch.no_grad():
            part = self.id - round
            part = part + NUM_WORKER if part < 0 else part
            part = part if part < NUM_WORKER else part - NUM_WORKER
            return_list = []
            for i in self.segmentation_result[part]:
                return_list.append(self.aggregated_grad[i])
            return return_list

    def update_model_scatter_receive(self, round, receive_model):
        """
        The round th step of the scatter stage

        :param round: this is the round th step
        :param receive_model: the received model
        """
        with torch.no_grad():
            part = self.id - round - 1
            part = part + NUM_WORKER if part < 0 else part
            part = part if part < NUM_WORKER else part - NUM_WORKER
            for i in self.segmentation_result[part]:
                self.aggregated_grad[i] += receive_model[i - self.segmentation_result[part][0]]
            if round == (NUM_WORKER -2):
                for i in self.segmentation_result[part]:
                    self.parameter[i] = self.parameter[i] - LR * self.aggregated_grad[i] / NUM_WORKER

    def update_model_reduce_send(self, round):
        with torch.no_grad():
            part = self.id + 1 - round
            part = part + NUM_WORKER if part < 0 else part
            part = part if part < NUM_WORKER else part - NUM_WORKER
            return_list = []
            for i in self.segmentation_result[part]:
                return_list.append(self.parameter[i])
            return return_list

    def update_model_reduce_receive(self, round, receive_model):
        with torch.no_grad():
            part = self.id - round
            part = part + NUM_WORKER if part < 0 else part
            part = part if part < NUM_WORKER else part - NUM_WORKER
            for i in self.segmentation_result[part]:
                self.parameter[i] = receive_model[i - self.segmentation_result[part][0]]
            if round == NUM_WORKER -2:
                for new_model, model in zip(self.parameter, self.net.parameters()):
                    model.data = new_model

class Train_test_init(object):
    def __init__(self, path = '../data'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=False,transform=transform_train)

        testset = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)
        self.acc_list = []
        self.flag = 0
        self.flag_list = np.arange(50000)

    def test_model(self, net):
        class_correct = 0
        class_total = 0
        with torch.no_grad():
            net.eval()
            for data in self.testloader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(16):
                    class_correct += c[i].item()
                    class_total += 1
                net.train()
            self.acc_list.append(class_correct / class_total)
            print("accuracy", self.acc_list[-1])
            net.train()

    def step(self):
        input = []
        label = []
        data = []
        for i in range(BZ):
            inp, lab = self.trainset[self.flag_list[self.flag]]
            input.append(inp)
            label.append(lab)
            self.__update()
        input = torch.stack(input, 0)
        label = torch.tensor(label)
        return (input, label)

    def __update(self):
        self.flag = self.flag + 1 if self.flag + 1 < len(self.trainset) else 0
        if self.flag == 0:
            np.random.shuffle(self.flag_list)

if __name__ == "__main__":
    train_test_init = Train_test_init()
    net = Net()
    segmentation = show_model_detail(net)
    worker_list = []
    for i in range(NUM_WORKER):
        worker_list.append(Worker(segmentation,i))
    total_loss = 0.0
    for i in range(ITERATION):
        iteration_loss = 0
        for j in range(NUM_WORKER):
            loss = worker_list[j].model_step(train_test_init.step())
            iteration_loss += loss / NUM_WORKER
        total_loss += iteration_loss
        for p in range(NUM_WORKER - 1): #scatter stage
            for j in range(NUM_WORKER): #process for each worker in scatter stage
                j_ = j + 1 - NUM_WORKER if j + 1 >= NUM_WORKER else j + 1
                worker_list[j_].update_model_scatter_receive(p, worker_list[j].update_model_scatter_send(p))
        for p in range(NUM_WORKER - 1): #reduce stage
            for j in range(NUM_WORKER):
                j_ = j + 1 - NUM_WORKER if j + 1 >= NUM_WORKER else j + 1
                worker_list[j_].update_model_reduce_receive(p, worker_list[j].update_model_reduce_send(p))
        if (i+1) % SHOW_LOSS == 0:
            print(total_loss/SHOW_LOSS)
            total_loss = 0.0
        if (i+1) % SHOW_ACC == 0:
            train_test_init.test_model(worker_list[0].net)


