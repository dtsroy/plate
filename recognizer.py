import torch.nn as nn
import torch

BASIC = 16


class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.conv1 = nn.Conv2d(3, BASIC, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(BASIC, BASIC * 2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(BASIC * 2, BASIC * 4, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv4 = nn.Conv2d(BASIC * 4, BASIC * 8, kernel_size=3, stride=1, padding=1)
        # self.relu4 = nn.ReLU()
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.relu5 = nn.ReLU()

        self.relu4 = nn.ReLU()

        self.fc = nn.Linear(BASIC * 4 * 64 * 64, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.pool4(x)
        #
        # x = self.relu5(x)

        x = self.relu4(x)

        x = x.view(-1, BASIC * 4 * 64 * 64)
        x = self.fc(x)

        return x


#
# class SelfAttention(nn.Module):
#     def __init__(self, in_dim):
#         super(SelfAttention, self).__init__()
#         self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
#         self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
#         self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         # Query, Key, Value
#         query = self.query_conv(x).view(x.size(0), -1, x.size(2)*x.size(3))
#         key = self.key_conv(x).view(x.size(0), -1, x.size(2)*x.size(3))
#         energy = torch.bmm(query.permute(0, 2, 1), key)
#         attention = self.softmax(energy)
#         value = self.value_conv(x).view(x.size(0), -1, x.size(2)*x.size(3))
#
#         out = torch.bmm(value, attention.permute(0, 2, 1))
#         out = out.view(x.size())
#         return out
#
#
# class Recognizer(nn.Module):
#     def __init__(self):
#         super(Recognizer, self).__init__()
#         self.conv1 = nn.Conv2d(3, BASIC, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.sa1 = SelfAttention(BASIC)
#
#         self.conv2 = nn.Conv2d(BASIC, BASIC * 2, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.sa2 = SelfAttention(BASIC * 2)
#
#         self.conv3 = nn.Conv2d(BASIC * 2, BASIC * 4, kernel_size=3, stride=1, padding=1)
#         self.relu3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.sa3 = SelfAttention(BASIC * 4)
#
#         self.relu4 = nn.ReLU()
#         self.fc = nn.Linear(BASIC * 4 * 64 * 64, 4)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.sa1(x)
#
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         x = self.sa2(x)
#
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.pool3(x)
#         x = self.sa3(x)
#
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
