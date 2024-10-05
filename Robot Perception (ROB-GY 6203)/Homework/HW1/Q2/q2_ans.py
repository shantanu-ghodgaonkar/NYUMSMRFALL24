# Import PyTorch
import torch
from torch import nn
import torchvision
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from time import time
import matplotlib.pyplot as plt

print(
    f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
torch.manual_seed(42)

# Setup device-agnostic code
if torch.cuda.is_available():
    device = "cuda"
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = "cpu"
    torch.set_default_tensor_type(torch.FloatTensor)

print(f"Using device: {device}")

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor())

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor())

BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator(device="cuda") if device == "cuda" else None)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                             generator=torch.Generator(device="cuda") if device == "cuda" else None)


class FashionMNISTVAE(nn.Module):
    def __init__(self, input_shape=784, hidden_units=512, output_shape=128):
        super(FashionMNISTVAE, self).__init__()
        if ((input_shape % 2 != 0) | (hidden_units % 2 != 0) | (output_shape % 2 != 0)):
            raise ValueError("Ensure that all parameters are multiples of 2")
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=int(hidden_units/2)),
            nn.ReLU(),
            nn.Linear(in_features=int(hidden_units/2),
                      out_features=output_shape)
        )
        self.fc_mu = nn.Linear(output_shape, int(
            output_shape/4))   # Mean vector
        self.fc_logvar = nn.Linear(
            output_shape, int(output_shape/4))  # Log variance

        self.decoder = nn.Sequential(
            nn.Linear(in_features=int(output_shape/4),
                      out_features=int(hidden_units/2)),
            nn.ReLU(),
            nn.Linear(in_features=int(hidden_units/2),
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=input_shape),
            nn.Sigmoid()
        )

    def encoder_func(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder_func(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder_func(x=x)
        z = self.reparametrization_trick(mu=mu, logvar=logvar)
        return self.decoder(z), mu, logvar


def FashionMNISTVAE_loss_func(org, recon, mu, logvar):
    # Reconstruction loss (binary cross entropy)
    reconstruction_loss = nn.functional.binary_cross_entropy(
        recon.view(-1, 28 * 28), org.view(-1, 28 * 28), reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - (mu ** 2) - torch.exp(logvar))

    return reconstruction_loss + kl_loss


def plot_reconstructed_images(model, n_images=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_images, 32)  # Sample from latent space
        generated = model.decoder_func(z)
        generated = generated.view(-1, 1, 28, 28)
        if device == 'cuda':
            generated = generated.to('cpu')

    fig, axes = plt.subplots(1, n_images, figsize=(n_images, 1))
    for i in range(n_images):
        axes[i].imshow(generated[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.waitforbuttonpress()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)  # Debugging for autograd
    model = FashionMNISTVAE(
        hidden_units=256, output_shape=16).to(device=device)
    # Smaller learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 50
    start_time = time()

    for epoch in range(epochs):
        model.train()
        tot_loss = 0
        tot_reconstruction_loss = 0
        tot_kl_loss = 0

        for data, _ in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(data)

            # Calculate individual components of the loss
            reconstruction_loss = nn.functional.binary_cross_entropy(
                recon.view(-1, 28 * 28), data.view(-1, 28 * 28), reduction='sum')
            kl_loss = -0.5 * \
                torch.sum(1 + logvar - (mu ** 2) - torch.exp(logvar))

            # Total loss
            loss = reconstruction_loss + kl_loss

            tot_loss += loss.item()
            tot_reconstruction_loss += reconstruction_loss.item()
            tot_kl_loss += kl_loss.item()

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {tot_loss / len(train_dataloader.dataset):.4f}, '
              f'Reconstruction Loss: {tot_reconstruction_loss / len(train_dataloader.dataset):.4f}, '
              f'KL Divergence Loss: {tot_kl_loss / len(train_dataloader.dataset):.4f}')

    print(f"Time taken for {epochs} epochs is {time() - start_time}s")
