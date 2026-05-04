import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics.nn.modules import C2PSA

class SceneClassifier(nn.Module):
    def __init__(self, backbone, num_classes=365, classes=None):
        super().__init__()

        backbone_layers = list(backbone[:9])
        backbone_layers.append(C2PSA(c1=256, c2=256, n=1))
        self.backbone = nn.ModuleList(backbone_layers)

        self.classes = classes

        for params in self.backbone.parameters():
            params.requires_grad = False

        for params in self.backbone[-1].parameters():
            params.requires_grad = True

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1280, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),

            nn.Dropout(p=0.3),
            nn.Linear(in_features=1280, out_features=512),
            nn.SiLU(),

            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        outputs = []
        for layer in self.backbone:
            if hasattr(layer, 'f') and layer.f != -1:
                if isinstance(layer.f, int):
                    x = outputs[layer.f]
                else:
                    x = [outputs[i] for i in layer.f]
            x = layer(x)
            outputs.append(x)
        return self.head(x)
    
    def predict(self, images: list[str], device: str=None, topk: int=5) -> list[dict]:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.eval()

        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        tensors = []
        for img in images:
            pil = Image.open(img).convert("RGB")
            tensors.append(transform(pil))

        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            logits = self(batch)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(topk, dim=-1)

        results = []
        for i in range(len(images)):
            top_entries = []
            for rank in range(topk):
                idx = top_indices[i, rank].item()
                conf = top_probs[i, rank].item()
                label = (self.classes[idx] if self.classes and idx < len(self.classes) else str(idx))
                top_entries.append({"label": label, "confidence": round(conf, 4)})

            results.append({
                "top_label":      top_entries[0]["label"],
                "top_confidence": top_entries[0]["confidence"],
                "top_k":          top_entries,
            })

        return results
"""
    def predict(self, x, topk=5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            top = probs.topk(topk)

        results = []
        for prob, index in zip(top.values[0], top.indices[0]):
            label = self.classes[index] if self.classes else str(index.item())
            results.append({
                "label": label,
                "index": index.item(),
                "prob": round(prob.item(), 4)
            })

        return results
"""