
import torch
import torch.nn as nn

DEFAULT_MARGIN = 2.0
# Określa on minimalną odległość, jaką próbki z różnych klas powinny mieć od siebie, aby funkcja straty niekarała ich

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=DEFAULT_MARGIN):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.sqrt(torch.sum(torch.pow(output1 - output2, 2), dim=1))  # Odległość L2 (euklidesowa)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
