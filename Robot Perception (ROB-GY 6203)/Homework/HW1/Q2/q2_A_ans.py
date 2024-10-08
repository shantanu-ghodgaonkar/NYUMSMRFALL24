import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from time import time
from tqdm import tqdm  # For displaying progress bars during training and evaluation

# ----------------------------
# 1. Configuration and Setup
# ----------------------------

# Configuration Parameters
CONFIG = {
    # Sets the random seed for reproducibility. Using 42 as a conventional seed value.
    'seed': 42,
    # Number of samples processed before the model's internal parameters are updated.
    'batch_size': 32,
    # Total number of complete passes through the training dataset.
    'epochs': 150,
    # Controls how much to adjust the model's parameters with respect to the loss gradient.
    'learning_rate': 0.001,
    # Size of the input layer. For 28x28 images, it's 784.
    'input_shape': 28 * 28,
    # Number of neurons in the hidden layers of the encoder and decoder.
    'hidden_units': 512,
    # Size of the latent (compressed) representation.
    'latent_dim': 10,
    # Base directory path where trained models will be saved.
    'model_save_path': Path("./Q2/saved_models_us").absolute(),
    # Directory path where datasets will be stored/downloaded.
    'dataset_path': Path("./Q2/data").absolute(),
}

# Modify the model_save_path based on the hyperparameters used for the training session.
# This aids in identifying the saved model based on its configuration.
CONFIG['model_save_path'] = CONFIG['model_save_path'] / \
    f"v2_BSZ{CONFIG['batch_size']}_EPS{CONFIG['epochs']}_LR{str(CONFIG['learning_rate']).replace('.', '_')}_HUS{CONFIG['hidden_units']}_LDM{CONFIG['latent_dim']}"


def set_seed(seed: int):
    """
    Ensures that results are reproducible by setting seeds for various random number generators.

    Args:
        seed (int): The seed value to be set for reproducibility.
    """
    torch.manual_seed(
        seed)  # Sets the seed for generating random numbers on the CPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Sets the seed for all GPUs.
    np.random.seed(seed)  # Sets the seed for NumPy's random number generator.
    random.seed(seed)  # Sets the seed for Python's built-in random module.


# Apply the seed for reproducibility.
set_seed(CONFIG['seed'])

# Device Configuration: Determines whether to use the GPU (cuda) or CPU based on availability.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Outputs the device being used for computations.
print(f"Using device: {device}")

# Create the directory for saving models if it doesn't already exist.
CONFIG['model_save_path'].mkdir(parents=True, exist_ok=True)

# ----------------------------
# 2. Data Preparation
# ----------------------------

# Define a chain of transformations to be applied to the dataset.
# Currently, only one transformation is used: converting images to PyTorch tensors.
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the FashionMNIST training dataset.
train_dataset = datasets.FashionMNIST(
    root=CONFIG['dataset_path'],  # Directory where the dataset will be stored.
    train=True,                   # Specifies that this is the training set.
    # Downloads the dataset if not already present.
    download=True,
    # Applies the defined transformations to the data.
    transform=transform
)

# Load the FashionMNIST testing dataset.
test_dataset = datasets.FashionMNIST(
    root=CONFIG['dataset_path'],  # Directory where the dataset will be stored.
    train=False,                  # Specifies that this is the test set.
    # Downloads the dataset if not already present.
    download=True,
    # Applies the defined transformations to the data.
    transform=transform
)

# Create DataLoader for the training dataset.
train_loader = DataLoader(
    train_dataset,                 # The dataset to load data from.
    batch_size=CONFIG['batch_size'],  # Number of samples per batch.
    shuffle=True,                  # Shuffles data at every epoch.
    num_workers=2,                 # Number of subprocesses for data loading.
    # Speeds up data transfer to GPU if available.
    pin_memory=True if torch.cuda.is_available() else False
)

# Create DataLoader for the testing dataset.
test_loader = DataLoader(
    test_dataset,                  # The dataset to load data from.
    batch_size=CONFIG['batch_size'],  # Number of samples per batch.
    # Does not shuffle data to maintain consistency during evaluation.
    shuffle=False,
    num_workers=2,                 # Number of subprocesses for data loading.
    # Speeds up data transfer to GPU if available.
    pin_memory=True if torch.cuda.is_available() else False
)

# ----------------------------
# 3. Model Definition
# ----------------------------


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for the FashionMNIST dataset.

    This model comprises an encoder that maps input images to a latent distribution
    and a decoder that reconstructs images from sampled latent vectors.
    """

    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=32):
        """
        Initializes the VAE with encoder and decoder networks.

        Args:
            input_dim (int, optional): Size of the input layer. Defaults to 784 (28x28 images).
            hidden_dim (int, optional): Number of neurons in the hidden layers. Defaults to 512.
            latent_dim (int, optional): Size of the latent (compressed) representation. Defaults to 32.
        """
        super(VAE, self).__init__()  # Initializes the parent class (nn.Module).

        # --------------------
        # Encoder Architecture
        # --------------------
        self.encoder = nn.Sequential(
            # Fully connected layer transforming input to hidden dimension.
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # ReLU activation introducing non-linearity.
            # Reduces dimensionality by half.
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),  # ReLU activation introducing non-linearity.
        )
        # Linear layers to compute the mean and log-variance of the latent distribution.
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # --------------------
        # Decoder Architecture
        # --------------------
        self.decoder = nn.Sequential(
            # Expands the latent vector back to the hidden dimension.
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),  # ReLU activation introducing non-linearity.
            # Further expands to the hidden dimension.
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),  # ReLU activation introducing non-linearity.
            # Maps back to the original input dimension.
            nn.Linear(hidden_dim, input_dim),
            # Sigmoid activation ensuring output values are between 0 and 1, suitable for image pixel intensities.
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encodes the input into latent space parameters (mean and log-variance).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log-variance tensors of shape (batch_size, latent_dim).
        """
        h = self.encoder(
            x)          # Passes input through the encoder network.
        # Computes the mean of the latent distribution.
        mu = self.fc_mu(h)
        # Computes the log-variance of the latent distribution.
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from the latent distribution.

        This allows backpropagation through stochastic nodes by expressing the sampling process
        as a deterministic function of the mean and variance.

        Args:
            mu (torch.Tensor): Mean tensor of shape (batch_size, latent_dim).
            logvar (torch.Tensor): Log-variance tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Sampled latent vector of shape (batch_size, latent_dim).
        """
        std = torch.exp(0.5 *
                        logvar)          # Calculates the standard deviation.
        # Samples epsilon from a standard normal distribution.
        eps = torch.randn_like(std)
        # Returns the sampled latent vector.
        return mu + eps * std

    def decode(self, z):
        """
        Decodes the latent vector back to the input space to reconstruct the image.

        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed image tensor of shape (batch_size, input_dim).
        """
        return self.decoder(z)                 # Passes the latent vector through the decoder network.

    def forward(self, x):
        """
        Defines the forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Reconstructed image tensor of shape (batch_size, input_dim).
                - Mean tensor of the latent distribution.
                - Log-variance tensor of the latent distribution.
        """
        mu, logvar = self.encode(
            x)            # Encodes the input to obtain latent distribution parameters.
        # Samples a latent vector using the reparameterization trick.
        z = self.reparameterize(mu, logvar)
        # Decodes the latent vector to reconstruct the image.
        reconstructed = self.decode(z)
        # Returns the reconstructed image and latent parameters.
        return reconstructed, mu, logvar

# ----------------------------
# 4. Loss Function
# ----------------------------


def loss_function(recon_x, x, mu, logvar):
    """
    Computes the VAE loss as the sum of reconstruction loss and Kullback-Leibler (KL) divergence.

    Args:
        recon_x (torch.Tensor): Reconstructed output from the decoder of shape (batch_size, input_dim).
        x (torch.Tensor): Original input data of shape (batch_size, input_dim).
        mu (torch.Tensor): Mean tensor from the encoder of shape (batch_size, latent_dim).
        logvar (torch.Tensor): Log-variance tensor from the encoder of shape (batch_size, latent_dim).

    Returns:
        torch.Tensor: The total loss for the batch.
    """
    # Binary Cross-Entropy (BCE) loss measures how well the decoder reconstructs the input.
    BCE = nn.functional.binary_cross_entropy(
        # Sums the loss over all elements in the batch.
        recon_x, x, reduction='sum'
    )
    # Kullback-Leibler (KL) divergence measures how much the latent distribution deviates from a standard normal distribution.
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total loss is the sum of reconstruction loss and KL divergence.
    return BCE + KLD

# ----------------------------
# 5. Training and Evaluation
# ----------------------------


def train(model, dataloader, optimizer, epoch):
    """
    Defines the training loop for one epoch.

    Args:
        model (VAE): An instance of the VAE class.
        dataloader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        epoch (int): Current epoch number.

    Returns:
        float: The average loss for the current epoch.
    """
    model.train()                     # Sets the model to training mode.
    # Initializes the cumulative training loss.
    train_loss = 0
    # Initializes a progress bar to monitor training progress.
    progress_bar = tqdm(enumerate(dataloader), total=len(
        dataloader), desc=f"Epoch {epoch}")

    for batch_idx, (data, _) in progress_bar:
        # Flattens and moves data to the designated device.
        data = data.view(-1, CONFIG['input_shape']).to(device)
        # Resets gradients from the previous iteration.
        optimizer.zero_grad()
        # Performs a forward pass through the model.
        recon_batch, mu, logvar = model(data)
        # Computes the loss.
        loss = loss_function(recon_batch, data, mu, logvar)
        # Backpropagates the loss.
        loss.backward()
        # Accumulates the loss.
        train_loss += loss.item()
        # Updates the model parameters.
        optimizer.step()
        # Updates the progress bar with the current batch loss.
        progress_bar.set_postfix({'Loss': loss.item() / len(data)})

    # Calculates the average loss over the entire training dataset.
    avg_loss = train_loss / len(dataloader.dataset)
    # Prints the average loss for the epoch.
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
    # Returns the average loss for potential logging.
    return avg_loss


def test(model, dataloader, epoch):
    """
    Defines the evaluation loop on the test dataset.

    Args:
        model (VAE): An instance of the VAE class.
        dataloader (DataLoader): DataLoader for the test dataset.
        epoch (int): Current epoch number.

    Returns:
        float: The average loss for the test dataset.
    """
    model.eval()                     # Sets the model to evaluation mode.
    test_loss = 0                    # Initializes the cumulative test loss.
    with torch.no_grad():            # Disables gradient computation for efficiency.
        for data, _ in dataloader:
            # Flattens and moves data to the designated device.
            data = data.view(-1, CONFIG['input_shape']).to(device)
            # Performs a forward pass through the model.
            recon, mu, logvar = model(data)
            # Computes the loss.
            loss = loss_function(recon, data, mu, logvar)
            # Accumulates the loss.
            test_loss += loss.item()
    # Calculates the average loss over the entire test dataset.
    avg_loss = test_loss / len(dataloader.dataset)
    # Prints the average test loss.
    print(f"====> Test set loss: {avg_loss:.4f}")
    # Returns the average test loss for potential logging.
    return avg_loss

# ----------------------------
# 6. Visualization
# ----------------------------


def plot_reconstructed_images(model, n=10):
    """
    Plots 'n' reconstructed images generated by the VAE.

    Args:
        model (VAE): An instance of the VAE class.
        n (int, optional): Number of images to generate and plot. Defaults to 10.
    """
    model.eval()  # Sets the model to evaluation mode.
    with torch.no_grad():  # Disables gradient computation.
        # Samples 'n' random latent vectors from a standard normal distribution.
        z = torch.randn(n, CONFIG['latent_dim']).to(device)
        # Decodes the latent vectors to generate images and moves them to the CPU.
        samples = model.decode(z).cpu()
    # Reshapes the samples to image dimensions (1 channel, 28x28 pixels).
    samples = samples.view(-1, 1, 28, 28)
    # Creates a subplot grid to display the images in a single row.
    fig, axes = plt.subplots(1, n, figsize=(n, 1))
    for i in range(n):
        # Displays each image in grayscale.
        axes[i].imshow(samples[i].squeeze(), cmap='gray')
        axes[i].axis('off')  # Hides the axis for a cleaner look.
    plt.suptitle('Generated Images')  # Adds a title above the subplots.
    plt.show()  # Renders and displays the plot.


def plot_tsne(model, dataloader):
    """
    Plots a t-SNE visualization of the latent space with color-coded labels.

    Args:
        model (VAE): An instance of the VAE class.
        dataloader (DataLoader): DataLoader for the dataset to visualize.
    """
    model.eval()  # Sets the model to evaluation mode.
    latent_vectors = []  # Initializes a list to store latent vectors.
    labels_list = []     # Initializes a list to store corresponding labels.
    with torch.no_grad():  # Disables gradient computation.
        for data, labels in dataloader:
            # Flattens and moves data to the designated device.
            data = data.view(-1, CONFIG['input_shape']).to(device)
            # Encodes the data to obtain latent distribution parameters.
            mu, logvar = model.encode(data)
            # Samples latent vectors using the reparameterization trick.
            z = model.reparameterize(mu, logvar)
            # Appends the latent vectors to the list.
            latent_vectors.append(z.cpu().numpy())
            # Appends the labels to the list.
            labels_list.append(labels.numpy())
    # Concatenates all latent vectors into a single NumPy array.
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    # Concatenates all labels into a single NumPy array.
    labels = np.concatenate(labels_list, axis=0)

    # Applies t-SNE to reduce the dimensionality of the latent vectors to 2D for visualization.
    tsne = TSNE(n_components=2,
                random_state=CONFIG['seed'], perplexity=30, n_iter=1000)
    latent_2d = tsne.fit_transform(latent_vectors)

    # Plots the t-SNE results.
    plt.figure(figsize=(10, 8))  # Sets the figure size.
    scatter = plt.scatter(
        latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10
    )  # Creates a scatter plot with color-coded labels.
    # Adds a color bar indicating label classes.
    plt.colorbar(scatter, ticks=range(10))
    plt.title('t-SNE of VAE Latent Space')  # Sets the plot title.
    plt.xlabel('Component 1')  # Labels the x-axis.
    plt.ylabel('Component 2')  # Labels the y-axis.
    plt.show()  # Renders and displays the plot.


def plot_loss_curves(train_losses, test_losses):
    """
    Plots the training and test loss curves over epochs.

    Args:
        train_losses (List[float]): List of training losses per epoch.
        test_losses (List[float]): List of test losses per epoch.
    """
    plt.figure(figsize=(10, 5))  # Sets the figure size.
    # Plots the training loss curve.
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')    # Plots the test loss curve.
    plt.xlabel('Epochs')                        # Labels the x-axis.
    plt.ylabel('Loss')                          # Labels the y-axis.
    plt.title('Training and Test Loss Over Epochs')  # Sets the plot title.
    # Adds a legend to differentiate the curves.
    plt.legend()
    # Renders and displays the plot.
    plt.show()

# ----------------------------
# 7. Model Persistence
# ----------------------------


def save_model(model, epoch, path):
    """
    Saves the model's state dictionary along with the epoch number.

    Args:
        model (VAE): An instance of the VAE class.
        epoch (int): Current epoch number.
        path (Path): Directory path where the model's state dictionary will be saved.
    """
    torch.save({
        'epoch': epoch,                      # Saves the current epoch number.
        # Saves the model's state dictionary containing all learnable parameters.
        'model_state_dict': model.state_dict(),
    }, path)                                  # Specifies the file path for saving.
    # Notifies the user that the model has been saved.
    print(f"Model saved to {path}")


def load_model(path, model):
    """
    Loads the model's state dictionary from the specified path.

    Args:
        path (Path): Directory path from where to load the model's state dictionary.
        model (VAE): An instance of the VAE class to which the state dictionary will be loaded.

    Raises:
        FileNotFoundError: If the specified path does not exist.

    Returns:
        VAE: The model instance with loaded parameters.
    """
    if not os.path.exists(path):  # Checks if the specified path exists.
        # Raises an error if the path does not exist.
        raise FileNotFoundError(f"No model found at {path}")
    # Loads the checkpoint from the path, mapping it to the current device.
    checkpoint = torch.load(path, map_location=device)
    # Loads the state dictionary into the model.
    model.load_state_dict(checkpoint['model_state_dict'])
    # Notifies the user that the model has been loaded.
    print(f"Model loaded from {path}")
    return model  # Returns the model with loaded parameters.

# ----------------------------
# 8. Main Execution
# ----------------------------


if __name__ == "__main__":
    # Initialize the VAE model and move it to the designated device (GPU or CPU).
    model = VAE(
        # Sets the input dimension (e.g., 784 for 28x28 images).
        input_dim=CONFIG['input_shape'],
        # Sets the number of hidden units in the encoder and decoder.
        hidden_dim=CONFIG['hidden_units'],
        # Sets the size of the latent space.
        latent_dim=CONFIG['latent_dim']
    ).to(device)

    # Initialize the Adam optimizer with the model's parameters and the specified learning rate.
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Lists to store the training and test losses for each epoch.
    train_losses = []
    test_losses = []

    start_time = time()  # Records the start time of the training process.

    # Training Loop: Iterates over the number of epochs.
    for epoch in range(1, CONFIG['epochs'] + 1):
        # Trains the model for one epoch.
        train_loss = train(model, train_loader, optimizer, epoch)
        # Evaluates the model on the test dataset.
        test_loss = test(model, test_loader, epoch)
        # Appends the training loss for the epoch.
        train_losses.append(train_loss)
        # Appends the test loss for the epoch.
        test_losses.append(test_loss)

        # Save the model checkpoint every 10 epochs.
        if epoch % 10 == 0:
            # Defines the save path with epoch number.
            save_path = CONFIG['model_save_path'] / f"VAE_epoch_{epoch}.pth"
            # Saves the model's state dictionary.
            save_model(model, epoch, save_path)

    total_time = time() - start_time  # Calculates the total training time.
    # Prints the total training duration in minutes.
    print(f"Training completed in {total_time/60:.2f} minutes.")

    # Final Model Saving: Saves the model after completing all epochs.
    final_save_path = CONFIG['model_save_path'] / \
        f"VAE_final_epoch_{CONFIG['epochs']}.pth"
    # Saves the final model.
    save_model(model, CONFIG['epochs'], final_save_path)

    # Plot Reconstructed Images:
    # Uncomment the following lines to generate and visualize reconstructed images.
    # plot_reconstructed_images(model, n=10)
    # model = load_model(path=final_save_path, model=model)

    # Plot t-SNE of Latent Space: Visualizes the structure of the latent space.
    plot_tsne(model, test_loader)

    # Plot Loss Curves: Visualizes the training and test loss over epochs.
    plot_loss_curves(train_losses, test_losses)
