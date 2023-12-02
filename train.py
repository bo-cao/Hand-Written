import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import load_mnist
from network import LeNet

ROOT = "./"
BATCH_SIZE = 64
LR = 0.001
EPOCH = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set, test_set, train_loader, test_loader = load_mnist.get_train_test_from_MNIST(root=ROOT, batch_size=BATCH_SIZE)
images, labels = next(iter(train_loader))

model = LeNet().to(DEVICE) # 模型
criterion = nn.CrossEntropyLoss() # 损失函数使用交叉熵
optimizer = optim.Adam(model.parameters(), lr=LR) # 优化函数使用 Adam 自适应优化算法

model = model.to(DEVICE)

for epoch in range(EPOCH):
    sum_loss = 0.0
    for i, data in enumerate(train_loader):
        # inputs, labels = data
        # inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        optimizer.zero_grad()  # 将梯度归零
        outputs = model(inputs)  # 将数据传入网络进行前向运算
        loss = criterion(outputs, labels)  # 得到损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 通过梯度做一步参数更新
        sum_loss += loss.item()
        if i % 100 == 99:
            print("[%d, %d] loss: %.03f" % (epoch + 1, i + 1, sum_loss / 100))
            sum_loss = 0.0


model.eval() # 将模型变换为测试模式
correct = 0
total = 0
for data_test in test_loader:
    # images, labels = data_test
    # images, labels = Variable(images), Variable(labels)
    images, labels = data_test[0].to(DEVICE), data_test[1].to(DEVICE)
    output_test = model(images)
    _, predicted = torch.max(output_test, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()


print("Correct1: ", correct)
print("Test Accuracy: {0}".format(correct.item() / len(test_set)))

torch.save(model, "./model12.2.pkl")