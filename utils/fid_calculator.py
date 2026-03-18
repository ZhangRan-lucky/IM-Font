"""
FID (Fréchet Inception Distance) Calculator
Uses Inception V3 to compute FID between generated and real images
"""
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from pathlib import Path
import pickle


class FIDCalculator:
    """Calculate FID score between generated and real images"""

    def __init__(self, device, real_images_path=None, batch_size=2048, cache_stats=True):
        """
        Args:
            device: torch device
            real_images_path: Path to directory with real images
            batch_size: Number of images for FID computation
            cache_stats: Whether to cache real image statistics
        """
        self.device = device
        self.batch_size = batch_size
        self.cache_stats = cache_stats
        self.real_images_path = real_images_path

        # Load Inception V3 model
        print("Loading Inception V3 for FID calculation...")
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = torch.nn.Identity()  # Remove final FC layer
        self.inception_model.to(device)
        self.inception_model.eval()

        # Preprocessing for Inception
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Cache for real statistics
        self.real_mu = None
        self.real_sigma = None

        # Try to load cached stats
        if real_images_path and cache_stats:
            cache_path = Path(real_images_path).parent / 'fid_stats_cache.pkl'
            if cache_path.exists():
                print(f"Loading cached FID statistics from {cache_path}")
                try:
                    with open(cache_path, 'rb') as f:
                        cached = pickle.load(f)
                        self.real_mu = cached['mu']
                        self.real_sigma = cached['sigma']
                        print("✓ Loaded cached statistics")
                except Exception as e:
                    print(f"Warning: Could not load cache: {e}")

        # Compute real statistics if not cached
        if self.real_mu is None and real_images_path:
            print("Computing statistics for real images (this will take a while)...")
            self._compute_real_statistics(real_images_path, cache_path if cache_stats else None)

    def _compute_real_statistics(self, image_path, cache_path=None):
        """Compute mean and covariance of Inception features for real images"""
        # Load real images
        try:
            dataset = ImageFolder(image_path, transform=transforms.ToTensor())
            # Limit to batch_size images
            if len(dataset) > self.batch_size:
                indices = np.random.choice(len(dataset), self.batch_size, replace=False)
                dataset = torch.utils.data.Subset(dataset, indices)

            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

            features = []
            print(f"Extracting features from {len(dataset)} real images...")

            with torch.no_grad():
                for batch, _ in dataloader:
                    batch = batch.to(self.device)
                    # Normalize for Inception
                    batch = (batch + 1) / 2  # [-1, 1] -> [0, 1] if needed
                    batch = self.preprocess(batch)

                    # Extract features
                    feat = self.inception_model(batch)
                    features.append(feat.cpu().numpy())

            features = np.concatenate(features, axis=0)

            # Compute statistics
            self.real_mu = np.mean(features, axis=0)
            self.real_sigma = np.cov(features, rowvar=False)

            print(f"✓ Real image statistics computed (mu shape: {self.real_mu.shape})")

            # Cache statistics
            if cache_path:
                try:
                    cache_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump({
                            'mu': self.real_mu,
                            'sigma': self.real_sigma
                        }, f)
                    print(f"✓ Statistics cached to {cache_path}")
                except Exception as e:
                    print(f"Warning: Could not cache statistics: {e}")

        except Exception as e:
            print(f"Error computing real statistics: {e}")
            print("FID will not be available")

    @torch.no_grad()
    def extract_features(self, images):
        """Extract Inception features from images"""
        features = []
        batch_size = 32

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)

            # Normalize to [0, 1]
            batch = (batch + 1) / 2

            # Resize to 299x299 for Inception
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

            # Normalize for Inception
            batch = self.preprocess(batch)

            # Extract features
            feat = self.inception_model(batch)
            features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Fréchet distance between two multivariate Gaussians"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print(f"Warning: fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def compute_fid(self, generated_images):
        """
        Compute FID score for generated images

        Args:
            generated_images: Tensor of shape [N, C, H, W] in range [-1, 1]

        Returns:
            FID score (float), lower is better
        """
        if self.real_mu is None or self.real_sigma is None:
            print("Error: Real image statistics not available")
            print("Please provide real_images_path when initializing FIDCalculator")
            return None

        # Extract features from generated images
        gen_features = self.extract_features(generated_images)

        # Compute statistics
        gen_mu = np.mean(gen_features, axis=0)
        gen_sigma = np.cov(gen_features, rowvar=False)

        # Calculate FID
        fid_score = self.calculate_frechet_distance(
            self.real_mu, self.real_sigma,
            gen_mu, gen_sigma
        )

        return float(fid_score)


# Example usage
if __name__ == "__main__":
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fid_calc = FIDCalculator(
        device=device,
        real_images_path='/path/to/real/images',
        batch_size=2048
    )

    # Generate some fake images for testing
    fake_images = torch.randn(100, 3, 64, 64) * 0.5

    # Compute FID
    fid = fid_calc.compute_fid(fake_images)
    print(f"FID Score: {fid:.2f}")