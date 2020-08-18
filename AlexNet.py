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


# 2) 搭建网络模型
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
            # LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, groups=2), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
            # LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 由此从卷积变为全连接层
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=6*6*256, out_features=4096), 
            nn.ReLU(inplace=True), 
            nn.Dropout()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096), 
            nn.ReLU(inplace=True), 
            nn.Dropout()
        )
        self.layer8 = nn.Linear(in_features=4096, out_features=1000)
    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(-1, 6*6*256)
        x = self.layer8(self.layer7(self.layer6(x)))
        return x


# 3) 导入使用的数据集、网络结构、优化器、损失函数等


# 4) 训练模型


# 5) 保存模型结构参数


# 6) 加载模型并测试模型效果

