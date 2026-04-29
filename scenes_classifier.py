import torch.nn as nn

class SceneClassifier(nn.Module):
    def __init__(self, backbone, num_classes=365):
        super().__init__()
        self.backbone = backbone

        for params in self.backbone.parameters():
            params.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1280, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),

            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return self.head(features)