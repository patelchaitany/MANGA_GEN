import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import image_gradients

class GradientNormLoss(nn.Module):
    def __init__(self):
        super(GradientNormLoss, self).__init__()

    def forward(self, img1, img2):
        def get_gradient_norm(img):
            dy, dx = image_gradients(img)
            return torch.sqrt(torch.mean(dy**2 + dx**2))

        norm_grad1 = get_gradient_norm(img1)
        norm_grad2 = get_gradient_norm(img2)
        loss = F.l1_loss(norm_grad1, norm_grad2)
        return loss

def test_gradient_norm_loss():
    # Test on a batch of images
    batch_size = 2
    channels = 3
    height = 64
    width = 64
    img1_batch = torch.randn(batch_size, channels, height, width)
    img2_batch = torch.randn(batch_size, channels, height, width)

    loss_fn = GradientNormLoss()

    # Check if GPU is available and move tensors and model to GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        img1_batch = img1_batch.to(device)
        img2_batch = img2_batch.to(device)
        loss_fn = loss_fn.to(device)
        print("Testing on GPU...")
    else:
        device = torch.device('cpu')
        print("Testing on CPU...")

    loss = loss_fn(img1_batch, img2_batch)
    print("Gradient Norm Loss:", loss.item())

if __name__ == '__main__':
    test_gradient_norm_loss()
