import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np


# 定义数据集类，这里只是一个简单示例，实际中需要根据数据集格式自定义
from sklearn.model_selection import train_test_split


# 创建虚拟数据集类
class CustomDataset(Dataset):
    def __init__(self, num_samples=1000, transform=None):
        self.num_samples = num_samples
        self.transform = transform

        # 生成随机图像数据和标签
        self.images = [torch.randn(3, 224, 224) for _ in range(num_samples)]  # 3通道，224x224大小的图像
        self.labels = [np.random.randint(0, 10) for _ in range(num_samples)]  # 假设有10个类别

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 数据预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建虚拟数据集实例
dataset = CustomDataset(num_samples=1000, transform=transform)

# 划分训练集和测试集
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 打印训练集和测试集的数据示例
for images, labels in train_loader:
    print("Training batch of images shape:", images.shape)
    print("Training batch of labels:", labels)
    break  # 仅打印第一个批次数据示例

for images, labels in test_loader:
    print("Testing batch of images shape:", images.shape)
    print("Testing batch of labels:", labels)
    break  # 仅打印第一个批次数据示例


# 定义模型类
class AlexNetLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size, lstm_num_layers):
        super(AlexNetLSTM, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.features = self.alexnet.features
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pooling(x)
        x = torch.flatten(x, 1)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out


# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

train_data = CustomDataset("train_data_dir", transform=transform)  # 训练数据集目录
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 定义模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
lstm_hidden_size = 128
lstm_num_layers = 2
model = AlexNetLSTM(num_classes, lstm_hidden_size, lstm_num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# 输出最终准确率
print(f"Final Accuracy: {epoch_accuracy:.2f}%")
