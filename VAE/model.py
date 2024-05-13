import torch
from torch import nn, reshape
import numpy as np

from torch.nn import functional as F

class VAECnn(nn.Module):
    def __init__(self, args) -> None:
        image_size = 28
        super().__init__()
        self.latent_size = args.latent_size
        self.p = args.p
        self.lambda_kl = args.lambda_kl
        self.num_layers = args.num_layers
        self.base_channels = args.base_channels
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.encoder = EncoderCnn(image_size, 1, self.num_layers, self.latent_size, base_channels=self.base_channels).to(device)

        self.decoder = DecoderCnn(image_size, self.base_channels*2**(self.num_layers-1), self.num_layers, self.latent_size, self.encoder.meta_shape[-1],base_channels=self.base_channels).to(device)
        
    def loss_func(self, input: torch.Tensor, only_reconstruction:bool=False):
        if not only_reconstruction:
            if torch.rand(1) > self.p:
                only_reconstruction = True
        
        mu, std = self.encoder.encode(input)
        z = self.reparameterize(mu, std)
        x_hat = self.decoder.decode(z)
        # print(torch.max(x_hat), torch.max(input),torch.max(x_hat-input))
        reconst_loss = F.mse_loss(x_hat, input)
        kl_div = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1).mean()
        if not only_reconstruction:
            return reconst_loss + self.lambda_kl*kl_div, reconst_loss, kl_div
        else:
            return reconst_loss, reconst_loss, kl_div
    def reparameterize(self, mu:torch.Tensor, std:torch.Tensor):
        """
        mu: (N, latent_size)
        std: (N, latent_size)
        output: (N, latent_size)
        """
        eps = torch.randn_like(std).to(self.device)
        return mu + eps*std
    
    def sample(self, num_samples):
        z = torch.randn((num_samples, self.latent_size)).to(self.device)
        return self.decoder.decode(z)
    
    def generate(self, num_images):
        return self.sample(num_samples=num_images)
    
class EncoderCnn(nn.Module):
    def __init__(self, image_size:int, in_channels:int, num_layers:int, latent_size:int,base_channels:int=16) -> None:
        super().__init__()
        
        assert num_layers >= 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.latent_size = latent_size
    
        self.meta = torch.rand(size = (1,in_channels, image_size, image_size)).to(self.device)
    
        hidden_channels =[in_channels] + [base_channels*int(2**i) for i in range(num_layers)]
        
        self.layers = nn.ModuleList()
        if num_layers<=2:
            for i in range(len(hidden_channels)-1):
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1],kernel_size=3,stride=2,padding=1),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2)
                    ).to(self.device)
                )
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=hidden_channels[i+1], out_channels=hidden_channels[i+1],kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2)
                    ).to(self.device)
                )
        else:
            for i in range(2):
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1],kernel_size=3,stride=2,padding=1),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2)
                    ).to(self.device)
                )
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=hidden_channels[i+1], out_channels=hidden_channels[i+1],kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2)
                    ).to(self.device)
                )
            for i in range(2, len(hidden_channels)-1):
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1],kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2)
                    ).to(self.device)
                )
        # self.proj = nn.Linear(np.prod(self.meta.shape), latent_size)
        with torch.no_grad():
            # shape = []
            x = self.meta.clone()
            for i in range(len(self.layers)):
                x = self.layers[i].forward(x)
            self.meta_shape = x.shape
        
        self.mu_proj = nn.Linear(np.prod(self.meta_shape), self.latent_size)
        self.std_proj = nn.Linear(np.prod(self.meta_shape), self.latent_size)
        
        
    def encode(self, image:torch.Tensor):
        """
        image: (N,C,H,W)
        output: (N, latent_size), (N, latent_size)
        """
        # print(image.shape)
        for i in range(len(self.layers)):
            image = self.layers[i].forward(image)

        image = image.view(-1, np.prod(self.meta_shape))
        mu = self.mu_proj(image)
        std = self.std_proj(image)
        return mu, std
    
    
class DecoderCnn(nn.Module):
    def __init__(self,image_size:int, in_channels:int, num_layers:int, latent_size:int, feature_map_size:int, base_channels:int=16) -> None:
        super().__init__()
        
        assert num_layers >= 1
        
        self.image_size = image_size
        self.feature_map_size = feature_map_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.latent_size = latent_size
        hidden_channels =[1] + [base_channels*int(2**i) for i in range(num_layers)]
        hidden_channels = list(reversed(hidden_channels))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList()
        
        self.feature = nn.Linear(latent_size, self.feature_map_size*self.feature_map_size*in_channels)
        
        if num_layers<=2:
            for i in range(len(hidden_channels)-1):
                self.layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1],kernel_size=4,stride=2,padding=1,output_padding=0),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(in_channels=hidden_channels[i+1], out_channels=hidden_channels[i+1],kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2)
                    ).to(self.device)
                )
        else:
            for i in range(2):
                self.layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1],kernel_size=4,stride=2,padding=1,output_padding=0),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(in_channels=hidden_channels[i+1], out_channels=hidden_channels[i+1],kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2)
                    ).to(self.device)
                )
            for i in range(2, len(hidden_channels)-1):
                self.layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1],kernel_size=3,stride=1,padding=1,output_padding=0),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(in_channels=hidden_channels[i+1], out_channels=hidden_channels[i+1],kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(hidden_channels[i+1]),
                        nn.LeakyReLU(0.2)
                    ).to(self.device)
                )
        # for i in range(len(hidden_channels)-1):
        #     self.layers.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1],kernel_size=4,stride=2,padding=1,output_padding=0),
        #             nn.BatchNorm2d(hidden_channels[i+1]),
        #             nn.LeakyReLU(0.2),
        #             nn.Conv2d(in_channels=hidden_channels[i+1], out_channels=hidden_channels[i+1],kernel_size=3,stride=1,padding=1),
        #             nn.BatchNorm2d(hidden_channels[i+1]),
        #             nn.LeakyReLU(0.2)
        #         ).to(self.device)
        #     )
    def decode(self, z:torch.Tensor):
        """
        z: (N, latent_size)
        output: (N, C, H, W)
        """
        z = self.feature(z)

        z = z.view(z.shape[0], -1, self.feature_map_size, self.feature_map_size)
        for i in range(len(self.layers)):
            z = self.layers[i].forward(z)
        return z
    
    
    
class VAEMlp(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.latent_size = args.latent_size
        self.p = args.p
        self.lambda_kl = args.lambda_kl
        self.num_layers = args.num_layers
        self.hiddens = args.hiddens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = EncoderMlp(self.hiddens, self.latent_size, self.num_layers).to(self.device)
        self.decoder = DecoderMlp(self.hiddens, self.latent_size, self.num_layers).to(self.device)
    def loss_func(self, input: torch.Tensor, only_reconstruction:bool=False):
        if not only_reconstruction:
            if torch.rand(1) > self.p:
                only_reconstruction = True
        
        mu, std = self.encoder.encode(input)
        z = self.reparameterize(mu, std)
        x_hat = self.decoder.decode(z)
        # print(torch.max(x_hat), torch.max(input),torch.max(x_hat-input))
        reconst_loss = F.mse_loss(x_hat, input)
        kl_div = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1).mean()
        if not only_reconstruction:
            return reconst_loss + self.lambda_kl*kl_div, reconst_loss, kl_div
        else:
            return reconst_loss, reconst_loss, kl_div
    def reparameterize(self, mu:torch.Tensor, std:torch.Tensor):
        """
        mu: (N, latent_size)
        std: (N, latent_size)
        output: (N, latent_size)
        """
        eps = torch.randn_like(std).to(self.device)
        return mu + eps*std
    
    def sample(self, num_samples):
        z = torch.randn((num_samples, self.latent_size)).to(self.device)
        return self.decoder.decode(z)
    
    def generate(self, num_images):
        return self.sample(num_samples=num_images)

class EncoderMlp(nn.Module):
    def __init__(self, hiddens:int, latent_size:int, num_layers:int) -> None:
        super().__init__()
        # self.input_size = input_size
        self.hiddens = hiddens
        self.latent_size = latent_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(784, hiddens).to(self.device))
            # elif i==num_layers-1:
            #     self.layers.append(nn.Linear(hiddens, latent_size).to(self.device))
            else:
                self.layers.append(nn.Linear(hiddens, hiddens).to(self.device))
            self.layers.append(nn.LeakyReLU(0.1).to(self.device))
        
        
        self.mu_proj = nn.Linear(hiddens, latent_size).to(self.device)
        self.std_proj = nn.Linear(hiddens, latent_size).to(self.device)
        
    def encode(self, x:torch.Tensor):
        """
        x: (N, input_size)
        output: (N, latent_size), (N, latent_size)
        """
        x = x.view(x.shape[0], -1)
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        mu = self.mu_proj(x)
        std = self.std_proj(x)
        return mu, std
    
class DecoderMlp(nn.Module):
    def __init__(self, hiddens:int, latent_size:int, num_layers:int) -> None:
        super().__init__()
        self.hiddens = hiddens
        self.latent_size = latent_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(latent_size, hiddens).to(self.device))
            # elif i==num_layers-1:
            #     self.layers.append(nn.Linear(hiddens, latent_size).to(self.device))
            else:
                self.layers.append(nn.Linear(hiddens, hiddens).to(self.device))
            self.layers.append(nn.LeakyReLU(0.1).to(self.device))
        
        self.output = nn.Linear(hiddens, 784).to(self.device)
        
    def decode(self, z:torch.Tensor):
        """
        z: (N, latent_size)
        output: (N, input_size)
        """
        for i in range(len(self.layers)):
            z = self.layers[i].forward(z)
        return self.output(z).reshape(-1, 1, 28, 28)
    