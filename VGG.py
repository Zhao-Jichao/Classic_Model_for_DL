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
import torchvision.transforms as transforms
from PIL import Image


# 2) 搭建网络模型
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.layer11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.layer22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.layer32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.layer33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layer42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layer43 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layer52 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layer53 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.pool1(nn.ReLU(self.layer12(nn.ReLU(self.layer11(x)))))
        x = self.pool2(nn.ReLU(self.layer22(nn.ReLU(self.layer21(x)))))
        x = self.pool3(nn.ReLU(self.layer33(nn.ReLU(self.layer32(nn.ReLU(self.layer31(x)))))))
        x = self.pool4(nn.ReLU(self.layer43(nn.ReLU(self.layer42(nn.ReLU(self.layer41(x)))))))
        x = self.pool5(nn.ReLU(self.layer53(nn.ReLU(self.layer52(nn.ReLU(self.layer51(x)))))))
        x = x.view(-1,7*7*512)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


cfg = {'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.features = self._make_layers(cfg['vgg16'])
        print(self.features)
        self.classifier = nn.Linear(512,10) #主要是实现CIFAR10，不同的任务全连接的结构不同

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

    def _make_layers(self,cfg):
        layers = []
        in_channels = 3
        for x in cfg:  #遍历list
            if x == 'M':
                layers += [nn.MaxPool2d(2,2)]
            else:
                layers += [nn.Conv2d(in_channels,x,3,1,1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)] ##inplace为True，将会改变输入的数据 ，
                                            # 否则不会改变原输入，只会产生新的输出
                in_channels = x
        #print(layers)
        return nn.Sequential(*layers)


# 3) 导入使用的数据集、网络结构、优化器、损失函数等
img = Image.open('/home/jichao/PythonDLbasedonPytorch/ClassicModels/224x224forVGG.png')
vgg = VGG16()
transform = transforms.ToTensor()
img = transform(img)
img = img.unsqueeze(0)
out = vgg(img)
print(vgg)

# 4) 训练模型


# 5) 保存模型结构参数


# 6) 加载模型并测试模型效果


