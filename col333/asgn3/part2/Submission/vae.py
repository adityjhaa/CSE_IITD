import sys

import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import f1_score

# Setting global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(14)
np.random.seed(0)

class CustomDataset(Dataset):
    """
    A custom dataset class for handling image data and optional labels.

    Args:
        data (array-like): The input data, typically a numpy array or similar structure.
        labels (array-like, optional): The labels corresponding to the input data. Default is None.
        transform (callable, optional): A function/transform to apply to the images. Default is None.

    Attributes:
        data (torch.Tensor): The input data converted to a PyTorch tensor.
        labels (torch.Tensor or None): The labels converted to a PyTorch tensor, or None if not provided.
        transform (callable or None): The transform function to apply to the images.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves the image (and label, if available) at the specified index.
    """
    def __init__(self, data, labels=None, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the image (and label, if available) at the specified index.
        """
        image = self.data[idx]
        label = None if self.labels is None else self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if label is not None:
            return image, label
        return image

class Reshape(nn.Module):
    """
    A custom PyTorch module that reshapes the input tensor to a specified shape.

    Args:
        shape (tuple): The desired shape to which the input tensor will be reshaped.

    Methods:
        forward(x):
            Reshapes the input tensor x to the specified shape.

    Example:
        >>> reshape = Reshape(-1, 2, 2)
        >>> input_tensor = torch.randn(4, 4)
        >>> output_tensor = reshape(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([4, 2, 2])
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    """
    A custom neural network module that trims the input tensor to a specified shape.

    Args:
        shape (tuple): The desired shape to trim the input tensor to. The shape should be provided as a tuple of integers.

    Methods:
        forward(x):
            Trims the input tensor `x` to the specified shape along the last two dimensions.

    Example:
        >>> trim_layer = Trim(1, 28, 28)
        >>> trimmed_tensor = trim_layer(input_tensor)
    """
    def __init__(self, *shape):
        super(Trim, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x[..., :self.shape[1], :self.shape[2]]

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    Args:
        latent_dim (int): The dimensionality of the latent space.

    Methods:
        encode(x):
            Encodes the input tensor `x` into mean and log variance of the latent space.
        reparameterize(mu, logvar):
            Reparameterizes the latent space using the mean and log variance.
        decode(z):
            Decodes the latent space `z` back into the reconstructed input.
        forward(x):
            Performs the forward pass through the VAE, returning the reconstructed input, mean, and log variance.
    """
    def __init__(self, latent_dim: int) -> None:
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),

            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 512 * 7 * 7)

        self.decoder = nn.Sequential(
            Reshape(-1, 512, 7, 7),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),

            Trim(-1, 28, 28),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encodes the input tensor `x` into mean and log variance of the latent space.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            mu (torch.Tensor): The mean of the latent space.
            logvar (torch.Tensor): The log variance of the latent space.
        """
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterizes the latent space using the mean and log variance.

        Args:
            mu (torch.Tensor): The mean of the latent space.
            logvar (torch.Tensor): The log variance of the latent space.

        Returns:
            z (torch.Tensor): The reparameterized latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std).to(mu.device)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent space `z` back into the reconstructed input.

        Args:
            z (torch.Tensor): The latent space tensor.

        Returns:
            x (torch.Tensor): The reconstructed input tensor.
        """
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Performs the forward pass through the VAE, returning the reconstructed input, mean, and log variance.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            x_recon (torch.Tensor): The reconstructed input tensor.
            mu (torch.Tensor): The mean of the latent space.
            logvar (torch.Tensor): The log variance of the latent space.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class GMM:
    """
    A Gaussian Mixture Model (GMM) implementation for clustering data, 
    using Expectation-Maximization (EM).
    """
    def __init__(self, n_components: int, max_iter=1000, tol=None, eps = 1e-6):
        """
        Initialize the model parameters.
        
        Paramters:
            n_components (int): The number of clusters (Gaussian components).
            max_iter (int, optional): Maximum number of iterations for EM algorithm. Defaults to 1000.
            tol (float, optional): Convergence threshold for log-likelihood improvement. Defaults to None.
            eps (float, optional): Small constant to add to the diagonal of the covariance matrix. Defaults to 1e-6.
        Returns:
            None
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means = None
        self.covariances = None
        self.weights = None # Mixing coefficients (weights of each Gaussian)
        self.epsilon = eps

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        Initialize means, covariances, and mixing weights.
        
        Parameters:
            X (ndarray): Input data with shape (n_samples, n_features).
        Returns:
            None
        """
        n_samples, n_features = X.shape

        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = self.means if self.means is not None else X[indices]

        self.covariances = [np.eye(n_features) for _ in range(self.n_components)]

        self.weights = np.full(self.n_components, 1 / self.n_components)

    def _regularize_covariance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Add a small constant to the diagonal to prevent singularity.
        
        Parameters:
            cov_matrix (ndarray): covariance matrix
        Returns:
            ndarray: regularized cov matrix
        """
        return cov_matrix + np.eye(cov_matrix.shape[0]) * self.epsilon

    def _safe_exponent(self, exponent: np.ndarray, max_val=1e10) -> np.ndarray:
        """
        Clipping exponent value for neumerical stability.
        Parameters:
            exponent (ndarray): exponentiated cov matrix.

        Returns:
            ndarray: Clipped exponentiated cov matrix.
        """
        return np.clip(exponent, -max_val, max_val)        
    
    def _compute_multivariate_gaussian(self, X: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """
        Compute multivariate Gaussian probability for each sample in X.
        Parameters:
            X (ndarray): Data points, shape (n_samples, n_features).
            mean (ndarray): Mean vector of a Gaussian component, shape (n_features,).
            covariance (ndarray): Covariance matrix of a Gaussian component, shape (n_features, n_features).

        Returns:
            ndarray: Gaussian probabilities, shape (n_samples,).
        """
        X = X.reshape(X.shape[0], -1)
        n, n_features = X.shape
        
        # Regularize the covariance matrix to avoid singularity
        cov = self._regularize_covariance(covariance)
        
        try:
            # Compute the determinant and inverse of the covariance matrix
            det_cov = np.linalg.det(cov)
            inv_cov = np.linalg.inv(cov)
    
            # Avoid invalid determinant values
            if det_cov <= 0:
                raise ValueError("Covariance matrix is singular.")
            
            # Compute the normalization factor
            norm_factor = 1.0 / (np.sqrt((2* np.pi) ** n_features * det_cov))
            
            # Calculate exponent safely
            diff = X - mean
            exponent_term = -0.5 * np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
            
            return norm_factor * np.exp(exponent_term)
        
        except np.linalg.LinAlgError:
            print("Error: Covariance matrix is not invertible.")
            return  np.full(X.shape[0], np.nan)  # Assign NaN to indicate failure in computation

    def _expectation_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-step: Calculate responsibility matrix.
        Parameters:
            X (ndarray): Input data, shape (n_samples, n_features).

        Returns:
            ndarray: Responsibility matrix, shape (n_samples, n_components).
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            MVG = self._compute_multivariate_gaussian(X, self.means[k], self.covariances[k])
            responsibilities[:, k] = self.weights[k] * MVG

        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= responsibilities_sum

        return responsibilities

    def _maximization_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        M-step: Update model parameters based on responsibilities.
        Parameters:
            X (ndarray): Input data, shape (n_samples, n_features).
            responsibilities (ndarray): Responsibility matrix.
        Returns:
            None
        """
        n_samples, _ = X.shape

        for k in range(self.n_components):
            N_k = responsibilities[:, k].sum()

            # Update means
            self.means[k] = (1 / N_k) * np.sum(responsibilities[:, k, None] * X, axis=0)

            # Update covariances
            diff = X - self.means[k]
            cov_k = (responsibilities[:, k, None] * diff).T @ diff / N_k
            self.covariances[k] = self._regularize_covariance(cov_k)

            # Update weights
            self.weights[k] = N_k / n_samples

    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute the log-likelihood of the data given the current model parameters.
        Parameters:
            X (ndarray): Input data, shape (n_samples, n_features).

        Returns:
            float: Log-likelihood of the data
        """
        log_likelihood = 0
        for k in range(self.n_components):
            MVG = self._compute_multivariate_gaussian(X, self.means[k], self.covariances[k])
            log_likelihood += self.weights[k] * MVG

        return np.sum(np.log(log_likelihood + 1e-10))

    def fit(self, X: np.ndarray) -> bool:
        """
        Fit the GMM to the data using the EM algorithm
        Parameters:
            X (ndarray): Training data, shape (n_samples, n_features).
        Returns:
            bool: True if the model converged, False otherwise
        """
        self._initialize_parameters(X)

        log_likelihood_prev = None
        
        for _ in range(self.max_iter):
            responsibilities = self._expectation_step(X)
            self._maximization_step(X, responsibilities)

            log_likelihood = self._compute_log_likelihood(X)
            if log_likelihood_prev is not None and self.tol is not None and abs(log_likelihood - log_likelihood_prev) < self.tol:
                return True
            log_likelihood_prev = log_likelihood
        
        return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the Gaussian component with the highest responsibility.
        Parameters:
            X (ndarray): Data to predict, shape (n_samples, n_features).

        Returns:
            list: Component labels for each sample.
        """
        responsibilities = self._expectation_step(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the probability of each sample for each component.
        Parameters:
            X (ndarray): Data to predict, shape (n_samples, n_features).

        Returns:
            ndarray: Probability matrix, shape (n_samples, n_components).
        """
        return self._expectation_step(X)   

def loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float=1) -> torch.Tensor:
    """
    Computes the loss function for the VAE, which includes the reconstruction loss and the KL divergence loss.

    Args:
        recon_x (torch.Tensor): The reconstructed input tensor.
        x (torch.Tensor): The original input tensor.
        mu (torch.Tensor): The mean of the latent space.
        logvar (torch.Tensor): The log variance of the latent space.
        beta (float, optional): The weight of the KL divergence loss. Default is 1.

    Returns:
        loss (torch.Tensor): The total loss, which is the sum of the reconstruction loss and the weighted KL divergence loss.
    """
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss

def train_vae(vae: VAE, train_loader: DataLoader, epoch: int, optimizer: torch.optim, beta_start=1, beta_end=1, num_epochs=100) -> float:
    vae.train()
    
    beta = beta_start + (beta_end - beta_start) * epoch / num_epochs
    
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar, beta)
        loss.backward()
        
        optimizer.step()

    return loss.item()

def initialize_gmm_means(vae: VAE, val_loader: DataLoader, device='cpu') -> tuple:
    """
    Calculate class-specific means in the latent space and directly set these as initial GMM cluster centers.

    :param vae: Trained VAE model (torch.nn.Module).
    :param val_loader: DataLoader - Validation data loader with images and labels.
    :param device: str - Device ('cpu' or 'cuda').
    :return: tuple - (initial_means, class_labels), where initial_means is the array of class-specific means 
             and class_labels contains the actual labels for each mean.
    """
    vae.eval()
    class_latent_vectors = {}

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            # Encode and extract mean latent vectors
            mu = vae.encode(images)[0].cpu().numpy()
            
            labels = labels.cpu().numpy()
            
            # Aggregate latent vectors by class label
            for mean, label in zip(mu, labels):
                if label not in class_latent_vectors:
                    class_latent_vectors[label] = []
                class_latent_vectors[label].append(mean)

    # Calculate mean vector for each class and store labels
    class_means = [np.mean(latent_vectors, axis=0) for latent_vectors in class_latent_vectors.values()]
    class_labels = list(class_latent_vectors.keys())
    
    return np.array(class_means), class_labels

def train_gmm_with_class_means(gmm: GMM, vae: VAE, train_loader: DataLoader, init_means: np.ndarray, device='cpu') -> bool:
    """
    Train the GMM on latent vectors with initial means set to class-specific means.
    
    :param gmm: GMM instance.
    :param vae: Trained VAE model (torch.nn.Module).
    :param train_loader: DataLoader - Training data loader with images.
    :param init_means: ndarray - Initial class-specific mean vectors to set GMM centers.
    :param device: str - Device ('cpu' or 'cuda')
    :return: bool - True if GMM converged, False otherwise.
    """
    vae.eval()
    
    # Set initial means in GMM directly as calculated from classes
    gmm.means = init_means
    
    # Extract latent vectors from training data
    latent_vectors = []
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            mean , log_var = vae.encode(images)

            z = vae.reparameterize(mean, log_var).cpu().numpy()
            latent_vectors.append(z)

    # Concatenate all latent vectors from mini-batches
    latent_vectors = np.vstack(latent_vectors)
    
    # Fit GMM to latent vectors
    return gmm.fit(latent_vectors)

def classify_images(vae: VAE, gmm: GMM, test_loader: DataLoader, class_labels: list, device='cpu') -> list:
    """
    Classify images by encoding them with the VAE, using the trained GMM, 
    and mapping clusters to actual class labels directly.
    
    :param vae: Trained VAE model.
    :param gmm: Trained GMM model.
    :param test_loader: DataLoader - Test data loader with images.
    :param class_labels: list - List of actual class labels in the order of GMM clusters.
    :param device: str - Device ('cpu' or 'cuda').
    :return: list - Predicted class labels for each sample.
    """
    vae.eval()
    predictions = []

    try:
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                mean = vae.encode(images)[0].cpu().numpy()

                # Predict cluster numbers using GMM and map directly to class labels
                clusters = gmm.predict(mean)
                batch_predictions = [class_labels[cluster] for cluster in clusters]
                predictions.extend(batch_predictions)
    except:
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                mean = vae.encode(images)[0].cpu().numpy()

                # Predict cluster numbers using GMM and map directly to class labels
                clusters = gmm.predict(mean)
                batch_predictions = [class_labels[cluster] for cluster in clusters]
                predictions.extend(batch_predictions)
        
    return predictions

def vae_recon(path_to_test_dataset_recon, vaePath):
    # Load the test dataset
    test_set = np.load(path_to_test_dataset_recon)
    
    # Transform the data into PyTorch tensors
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x),
        transforms.Lambda(lambda x: x.unsqueeze(0) if len(x.shape) == 2 else x),
        transforms.Normalize(
            mean=(0,),
            std=(255,)
        )
    ])
    
    # Create the Dataset and DataLoader objects for the test dataset
    test_data = CustomDataset(test_set['data'], transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Load the VAE model
    LATENT_DIM = 2
    vae = VAE(LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(vaePath, map_location=device))
    
    # Set the model to evaluation mode
    vae.eval()
    
    # Initialize lists to store the original and reconstructed images
    reconstructed_images = []
    try:
        with torch.no_grad():
            for image, _ in test_loader:
                image = image.to(device)
                recon_image, _, _ = vae(image)

                recon_image = recon_image.cpu().squeeze().numpy()
                
                reconstructed_images.append(recon_image)
    except:
        with torch.no_grad():
            for image in test_loader:
                image = image.to(device)
                recon_image, _, _ = vae(image)

                recon_image = recon_image.cpu().squeeze().numpy()
                
                reconstructed_images.append(recon_image)
    
    # Save image as npz file
    reconstructed_images = np.array(reconstructed_images)
    np.savez('vae_reconstructed.npz', data=reconstructed_images)

def class_pred(path_to_test_dataset, vaePath, gmmPath):
    # Load the test dataset
    test_set = np.load(path_to_test_dataset)
    
    # Transform the data into PyTorch tensors
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x),
        transforms.Lambda(lambda x: x.unsqueeze(0) if len(x.shape) == 2 else x),
        transforms.Normalize(
            mean=(0,),
            std=(255,)
        )
    ])
    
    # Create the Dataset and DataLoader objects for the test dataset
    test_data = CustomDataset(test_set['data'], transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Load the VAE model
    LATENT_DIM = 2
    vae = VAE(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(vaePath, map_location=device))
    
    # Set the model to evaluation mode
    vae.eval()
    
    # Load the GMM Model
    with open(gmmPath, 'rb') as f:
        gmm = pickle.load(f)
    
    predictions = classify_images(vae, gmm, test_loader, [1, 4, 8], device=device)
    
    # Save as csv
    df = pd.DataFrame(predictions, columns=['Predicted_Label'])
    df.to_csv("vae.csv", index=False)

def training_main(path_to_train_dataset, path_to_val_dataset, vaePath, gmmPath, verbose=True):
    
    # ------------------- Training the VAE ------------------- #
    
    # Setting the hyperparameters for VAE training
    LATENT_DIM = 2
    NUM_EPOCHS = 100
    BETA_START = 1
    BETA_END = 2
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4
    ETA_MIN = 1e-6
    
    # Setting hyperparameters for GMM training
    MAX_ITER = 100
    TOL = 1e-6
    N_COMPONENTS = 3
    EPS = 1e-6
    
    # Load the training and validation datasets
    train_set = np.load(path_to_train_dataset)
    val_set = np.load(path_to_val_dataset)
    
    # Transform the data into PyTorch tensors
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x),
        transforms.Lambda(lambda x: x.unsqueeze(0) if len(x.shape) == 2 else x),
        transforms.Normalize(
            mean=(0,),
            std=(255,)
        )
        ])
    
    # Create a custom dataset class for the training and validation datasets
    train_data = CustomDataset(train_set['data'], train_set['labels'], transform=transform)
    val_data = CustomDataset(val_set['data'], val_set['labels'], transform=transform)
    
    # Create DataLoader objects for the training and validation datasets
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize the VAE model
    vae = VAE(LATENT_DIM).to(device)
    
    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)
    
    # Train the VAE model
    for epoch in range(NUM_EPOCHS):
        train_loss = train_vae(vae, train_loader, epoch, optimizer, BETA_START, BETA_END, NUM_EPOCHS)
        scheduler.step()
        
        if verbose:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss:.4f}")
        
    # Save the trained VAE model
    torch.save(vae.state_dict(), vaePath)
    
    # ------------------- Training the GMM ------------------- #
    f1_prev = -1
    gmm_final = None
    
    for _ in range(1, MAX_ITER+1):
        initial_means, class_labels = initialize_gmm_means(vae, val_loader, device)
        
        gmm = GMM(n_components=N_COMPONENTS, max_iter=MAX_ITER, tol=TOL, eps=EPS)
        _ = train_gmm_with_class_means(gmm, vae, val_loader, initial_means, device)
        
        predictions = classify_images(vae, gmm, val_loader, class_labels, device)
        
        f1 = (f1_score(val_set['labels'], predictions, average='macro') + f1_score(val_set['labels'], predictions, average='micro')) / 2
        
        if f1 >= f1_prev:
            f1_prev = f1
            if verbose:
                print(f"Iteration {_}/{MAX_ITER}, F1 Score: {f1:.4f}")
            gmm_final = gmm
        else:
            break
        
    # Save the trained GMM model as a pkl file
    with open(gmmPath, 'wb') as f:
        pickle.dump(gmm_final, f)

if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None

    if len(sys.argv) == 4:  # Running code for VAE reconstruction.
        path_to_test_dataset_recon = arg1
        test_reconstruction = arg2
        vaePath = arg3
        vae_recon(arg1, arg3)

    elif len(sys.argv) == 5:  # Running code for class prediction during testing
        path_to_test_dataset = arg1
        test_classifier = arg2
        vaePath = arg3
        gmmPath = arg4
        class_pred(arg1, arg3, arg4)

    else:  # Running code for training. Save the model in the same directory with name "vae.pth"
        path_to_train_dataset = arg1
        path_to_val_dataset = arg2
        trainStatus = arg3
        vaePath = arg4
        gmmPath = arg5
        training_main(arg1, arg2, arg4, arg5, verbose=False)