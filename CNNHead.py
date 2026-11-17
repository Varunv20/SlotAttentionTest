import torch
import torch.nn as nn
class CNNEncoder(nn.Module):
    def __init__(self, hidden_dim=64, image_size=128):
        super().__init__()
        self.output_dim = hidden_dim   # NEW: SlotAttentionAutoencoder reads this to size slot attention's input projection

        self.conv1 = nn.Conv2d(3, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

        num_tokens = image_size * image_size
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode(self, images):
        x = self.relu(self.conv1(images))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        # x: [batch, hidden_dim, H, W]

        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1)       
        x = x.reshape(B, H * W, D)       


        x = x + self.pos_embed

        x = self.norm(x)
        x = x + self.mlp(x)
        return x


