import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, confidence=10):
        super(MLP, self).__init__()
        self.confidence = confidence
        self.fc1 = nn.Linear(self.confidence, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # x = x.view(-1, self.num_classes)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class MLP(nn.Module):
#     def __init__(self, num_classes=10):
#         super(MLP, self).__init__()
#         self.num_classes = num_classes
#         self.fc1 = nn.Linear(self.num_classes, 2)

#     def forward(self, x):
#         # x = x.view(-1, self.num_classes)
#         x = self.fc1(x)
#         return x

# class MLP(nn.Module):
#     def __init__(self, latent_feature=512):
#         super(MLP, self).__init__()
#         self.latent_feature = latent_feature
#         self.fc1 = nn.Linear(self.latent_feature, 128)
#         self.fc2 = nn.Linear(128, 16)
#         self.fc3 = nn.Linear(16, 2)

#     def forward(self, x):
#         # x = x.view(-1, self.num_classes)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
