import torch
import torch.nn as nn


class SEBlock3D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        hidden = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class SEBlock2D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        hidden = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Backbone(nn.Module):
    def __init__(self, in_chs, patch_size, dataset=None):
        super().__init__()
        dataset = dataset or ("houston" if in_chs == 48 else "pavia")
        houston = dataset == "houston"
        self.in_chs = in_chs
        self.patch_size = patch_size

        width3d = 12 if houston else 16
        first2d = 24 if houston else 32
        width2d = 32 if houston else 64
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(7, 3, 3), stride=(2, 1, 1)),
            nn.GroupNorm(2, 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(4, 8, kernel_size=(5, 3, 3), stride=(2, 1, 1)),
            nn.GroupNorm(2, 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, width3d, kernel_size=(3, 3, 3)),
            nn.GroupNorm(3 if houston else 4, width3d),
            nn.ReLU(inplace=True),
            SEBlock3D(width3d),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_chs, first2d, kernel_size=1),
            nn.GroupNorm(6 if houston else 8, first2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(first2d, width2d, kernel_size=3),
            nn.GroupNorm(8 if houston else 16, width2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(width2d, width2d, kernel_size=3),
            nn.GroupNorm(8 if houston else 16, width2d),
            nn.ReLU(inplace=True),
            SEBlock2D(width2d),
        )

        self.dim3d, self.dim2d = self._get_flatten_shapes()
        self.out_dim = self.dim3d + self.dim2d

    def _get_flatten_shapes(self):
        x2d = torch.zeros((1, self.in_chs, self.patch_size, self.patch_size))
        x3d = x2d.unsqueeze(1)
        with torch.no_grad():
            out3d = self.conv3d(x3d)
            out2d = self.conv2d(x2d)
        return out3d[0].numel(), out2d[0].numel()

    def get_out_dim(self):
        return self.out_dim

    def forward(self, x):
        y3d = self.conv3d(x.unsqueeze(1)).flatten(1)
        y2d = self.conv2d(x).flatten(1)
        return torch.cat([y3d, y2d], dim=1)


class AttentionParallelCNN(nn.Module):
    def __init__(self, in_chs, class_num, patch_size, dataset=None):
        super().__init__()
        self.in_chs = in_chs
        dataset = dataset or ("houston" if in_chs == 48 else "pavia")
        houston = dataset == "houston"
        self.class_num = class_num
        self.patch_size = patch_size

        self.backbone = Backbone(in_chs, patch_size, dataset)
        self.inter_dim = self.backbone.get_out_dim()
        if houston:
            layers = [nn.Linear(self.inter_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.35),
                      nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(0.25)]
        else:
            layers = [nn.Linear(self.inter_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
                      nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.4),
                      nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(0.4)]
        self.fc = nn.Sequential(*layers)
        self.classifier = nn.Linear(32, class_num)
        self.feature_dim = 32

    def extract_features(self, x):
        return self.fc(self.backbone(x))

    def forward(self, x):
        feature = self.extract_features(x)
        return self.classifier(feature), feature
