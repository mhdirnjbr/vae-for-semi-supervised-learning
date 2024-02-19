import torch
import torch.nn as nn
import torch.nn.functional as F

           
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class FashionMNISTEncoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        hidden_dim=256
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), 
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), 
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.ReLU(True),
            nn.Conv2d(128, hidden_dim, 4, 1),
            nn.ReLU(True),
            View((-1, hidden_dim*1*1)),
            nn.Linear(hidden_dim, z_dim*2),
        )

        self.locs = nn.Linear(hidden_dim, z_dim)
        self.scales = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = self.encoder(x)
        return hidden[:, :self.z_dim], torch.clamp(F.softplus(hidden[:, self.z_dim:]), min=1e-3)
     
        
class FashionMNISTDecoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        hidden_dim=256
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),  
            View((-1, hidden_dim, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 128, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)

class Classifier(nn.Module):
    def __init__(self, z_dim, classes):
        super(Classifier, self).__init__()
        h_dim = 500
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, classes)
        )

    def forward(self, x):
        return self.classifier(x)

class ConditionalPrior(nn.Module):
    def __init__(self, z_dim, classes):
        super(ConditionalPrior, self).__init__()
        h_dim = 500
        self.locs = nn.Linear(classes, z_dim)
        self.scales = nn.Linear(classes, z_dim)

    def forward(self, x):
        return self.locs(x), torch.clamp(F.softplus(self.scales(x)), min=1e-3)