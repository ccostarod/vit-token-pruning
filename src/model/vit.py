import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def create_vit_model(num_classes: int = 102, pretrained: bool = True):
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
    else:
        weights = None

    model = vit_b_16(weights=weights)
    
    # Substituir a cabeça de classificação para o número correto de classes
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model
