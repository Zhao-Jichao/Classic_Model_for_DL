# 1) 导入必需的包
# 2) 搭建网络模型
# 3) 导入使用的数据集、网络结构、优化器、损失函数等
# 4) 训练模型
# 5) 保存模型结构参数
# 6) 加载模型并测试模型效果

# 1) 导入必需的包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


# 2) 搭建网络模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 遇到了卷积层变为全连接层
        self.c5 = nn.Linear(16*5*5, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.c1(x)
        # x = F.relu(x)
        x = self.s2(x)
        x = self.c3(x)
        # x = F.relu(x)
        x = self.s4(x)
        # 刚才的卷积层变为全连接层，使用下边代码实现
        x = x.view(-1, 16*5*5)
        x = self.c5(x)
        # x = F.relu(x)
        x = self.f6(x)
        # x = F.relu(x)
        x = self.f7(x)
        # x = F.sigmoid(x)
        return x


# 3) 导入使用的数据集、网络结构、优化器、损失函数等
LeNet5 = LeNet5()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LeNet5 = LeNet5.to(device=device)

optimizer = optim.Adam(LeNet5.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
batch_size = 128
num_epoch = 2

transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)]
)
# 如果没有下载 MNIST 数据集，那么需要设置 download 参数为 True
# 如果已经下载 MNIST 数据集，那么只需设置 download 参数为 False
trainset = torchvision.datasets.MNIST(root='./data', 
                                    train=True, 
                                    transform=transform, 
                                    download=False)
trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', 
                                    train=False, 
                                    transform=transform, 
                                    download=False)
testloader = torch.utils.data.DataLoader(dataset=testset, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=2)


# 4) 训练模型
for epoch in range(num_epoch):
    LeNet5.train(0)
    train_loss = 0.0
    for idx, (img, label) in enumerate(trainloader):
        # print('开始训练第 {} 批数据'.format(idx+1))
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        output = LeNet5(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('完成第 {} 轮训练，train_loss: {}'.format(epoch+1, train_loss))


# 5) 保存模型结构参数
torch.save(LeNet5.state_dict(), 'LeNet5_state_dict.pth')

# 6) 加载模型并测试模型效果
LeNet5.load_state_dict(torch.load('LeNet5_state_dict.pth'))
LeNet5.eval()

correct_num = 0
data_num = 0
for idx, (img, label) in enumerate(testloader):
    img = img.to(device)
    label = label.to(device)
    output = LeNet5(img)
    idx, pred = output.max(1)
    for i in range(len(label)):
        if label[i] == pred[i]:
            correct_num += 1
        data_num += 1
    print('data_num: {} \t correct_num: {} \t Accuracy: {:.2f}%'.format(data_num, correct_num, correct_num*100/data_num))