import torch
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
import os
import glob
from scipy import linalg
import argparse
from datetime import datetime


def calculate_metrics(real_images, generated_images, device='cuda'):
    """Calculate SSIM, RMSE, L1, LPIPS metrics"""
    # SSIM
    ssim_values = []
    for real, gen in zip(real_images, generated_images):
        real_np = np.array(real)
        gen_np = np.array(gen)
        if real_np.shape != gen_np.shape:
            gen = gen.resize(real.size, Image.BICUBIC)
            gen_np = np.array(gen)
        ssim_values.append(ssim(real_np, gen_np, channel_axis=2, win_size=3))
    ssim_score = np.mean(ssim_values)

    # RMSE
    rmse_values = []
    for real, gen in zip(real_images, generated_images):
        real_np = np.array(real).astype(np.float32) / 255.0
        gen_np = np.array(gen)
        if gen_np.shape != real_np.shape:
            gen = gen.resize(real.size, Image.BICUBIC)
            gen_np = np.array(gen)
        gen_np = gen_np.astype(np.float32) / 255.0
        rmse_values.append(np.sqrt(mean_squared_error(real_np.flatten(), gen_np.flatten())))
    rmse_score = np.mean(rmse_values)

    # L1 (Mean Absolute Error)
    l1_values = []
    for real, gen in zip(real_images, generated_images):
        real_np = np.array(real).astype(np.float32) / 255.0
        gen_np = np.array(gen)
        if gen_np.shape != real_np.shape:
            gen = gen.resize(real.size, Image.BICUBIC)
            gen_np = np.array(gen)
        gen_np = gen_np.astype(np.float32) / 255.0
        l1_values.append(mean_absolute_error(real_np.flatten(), gen_np.flatten()))
    l1_score = np.mean(l1_values)

    # LPIPS
    lpips_metric = lpips.LPIPS(net='alex').to(device)
    lpips_values = []

    for real, gen in zip(real_images, generated_images):
        if gen.size != real.size:
            gen = gen.resize(real.size, Image.BICUBIC)

        real_tensor = transforms.ToTensor()(real).unsqueeze(0).to(device)
        gen_tensor = transforms.ToTensor()(gen).unsqueeze(0).to(device)
        real_tensor = real_tensor * 2 - 1
        gen_tensor = gen_tensor * 2 - 1

        with torch.no_grad():
            lpips_score = lpips_metric(real_tensor, gen_tensor)
        lpips_values.append(lpips_score.item())

    lpips_score = np.mean(lpips_values)

    return {
        'SSIM': ssim_score,
        'RMSE': rmse_score,
        'L1': l1_score,
        'LPIPS': lpips_score
    }


def load_inception_model(device='cuda'):
    """Load pretrained Inception V3 model"""
    print("  Loading Inception V3 model...")
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def extract_inception_features(images, model, device='cuda', batch_size=32):
    """Extract features using Inception V3"""
    features = []

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_tensors = torch.stack([preprocess(img) for img in batch]).to(device)

        with torch.no_grad():
            batch_features = model(batch_tensors)

        features.append(batch_features.cpu().numpy())

    return np.concatenate(features, axis=0)


def calculate_fid(real_features, gen_features):
    """Calculate Frechet Inception Distance (FID)"""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


def calculate_fid_for_folders(real_images, gen_images, inception_model, device='cuda'):
    """Calculate FID for a set of images"""
    real_features = extract_inception_features(real_images, inception_model, device)
    gen_features = extract_inception_features(gen_images, inception_model, device)
    fid_score = calculate_fid(real_features, gen_features)
    return fid_score


def natural_sort_key(path):
    """Extract numbers from filename for sorting"""
    import re
    filename = os.path.basename(path)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0


def evaluate_single_style(style_id, real_base_path, gen_base_path, inception_model, device):
    """Evaluate a single style"""
    real_folder = os.path.join(real_base_path, f'id_{style_id}')
    gen_folder = os.path.join(gen_base_path, f'id_{style_id}')

    if not os.path.exists(real_folder):
        print(f"  Warning: Real images folder not found: {real_folder}")
        return None

    if not os.path.exists(gen_folder):
        print(f"  Warning: Generated images folder not found: {gen_folder}")
        return None

    real_image_files = sorted(
        glob.glob(os.path.join(real_folder, '*.png')) +
        glob.glob(os.path.join(real_folder, '*.jpg')),
        key=natural_sort_key
    )
    gen_image_files = sorted(
        glob.glob(os.path.join(gen_folder, '*.png')) +
        glob.glob(os.path.join(gen_folder, '*.jpg')),
        key=natural_sort_key
    )

    if len(real_image_files) == 0 or len(gen_image_files) == 0:
        print(f"  Warning: No images found in id_{style_id}")
        return None

    min_images = min(len(real_image_files), len(gen_image_files))

    print(f"  Loading {min_images} image pairs...")
    try:
        real_images = [Image.open(f).convert('RGB') for f in real_image_files[:min_images]]
        generated_images = [Image.open(f).convert('RGB') for f in gen_image_files[:min_images]]
    except Exception as e:
        print(f"  Error loading images: {e}")
        return None

    print(f"  Calculating SSIM, RMSE, L1, LPIPS...")
    metrics = calculate_metrics(real_images, generated_images, device)

    print(f"  Calculating FID...")
    fid_score = calculate_fid_for_folders(real_images, generated_images, inception_model, device)
    metrics['FID'] = fid_score
    metrics['num_images'] = min_images

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation for multiple styles')
    parser.add_argument('--real_base_path', type=str,
                        default='/path/to/test',
                        help='Base path containing real image folders')
    parser.add_argument('--gen_base_path', type=str,
                        default='/path/to/results',
                        help='Base path containing generated image folders')
    parser.add_argument('--start_id', type=int, default=0,
                        help='Starting style ID (default: 0)')
    parser.add_argument('--end_id', type=int, default=19,
                        help='Ending style ID (default: 19)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path for results')
    args = parser.parse_args()

    real_base_path = args.real_base_path
    gen_base_path = args.gen_base_path
    start_id = args.start_id
    end_id = args.end_id
    output_file = args.output_file

    if not os.path.exists(real_base_path):
        print(f"Error: Real images base folder not found: {real_base_path}")
        return

    if not os.path.exists(gen_base_path):
        print(f"Error: Generated images base folder not found: {gen_base_path}")
        return

    if output_file is None:
        output_file = os.path.join(gen_base_path, f'evaluation.txt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    print("=" * 70)
    print("LOADING INCEPTION V3 MODEL")
    print("=" * 70)
    inception_model = load_inception_model(device)
    print("Model loaded successfully!\n")

    all_results = {}
    successful_evals = 0
    failed_evals = 0

    print("=" * 70)
    print(f"BATCH EVALUATION: id_{start_id} to id_{end_id}")
    print("=" * 70)

    for style_id in range(start_id, end_id + 1):
        print(f"\n{'-' * 70}")
        print(f"Evaluating id_{style_id} ({style_id - start_id + 1}/{end_id - start_id + 1})")
        print(f"{'-' * 70}")

        metrics = evaluate_single_style(style_id, real_base_path, gen_base_path, inception_model, device)

        if metrics is not None:
            all_results[style_id] = metrics
            successful_evals += 1
            print(f"id_{style_id} completed:")
            print(f"   SSIM: {metrics['SSIM']:.4f} | RMSE: {metrics['RMSE']:.4f} | L1: {metrics['L1']:.4f}")
            print(f"   LPIPS: {metrics['LPIPS']:.4f} | FID: {metrics['FID']:.4f}")
        else:
            failed_evals += 1
            print(f"id_{style_id} failed")

    if successful_evals > 0:
        avg_metrics = {
            'SSIM': np.mean([r['SSIM'] for r in all_results.values()]),
            'RMSE': np.mean([r['RMSE'] for r in all_results.values()]),
            'L1': np.mean([r['L1'] for r in all_results.values()]),
            'LPIPS': np.mean([r['LPIPS'] for r in all_results.values()]),
            'FID': np.mean([r['FID'] for r in all_results.values()])
        }

        print(f"\n{'=' * 70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total styles evaluated: {successful_evals}/{end_id - start_id + 1}")
        print(f"Failed evaluations: {failed_evals}")
        print(f"\nAverage Metrics (across all styles):")
        print(f"  SSIM:  {avg_metrics['SSIM']:.4f} (higher is better)")
        print(f"  RMSE:  {avg_metrics['RMSE']:.4f} (lower is better)")
        print(f"  L1:    {avg_metrics['L1']:.4f} (lower is better)")
        print(f"  LPIPS: {avg_metrics['LPIPS']:.4f} (lower is better)")
        print(f"  FID:   {avg_metrics['FID']:.4f} (lower is better)")
        print(f"{'=' * 70}\n")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 70}\n")
            f.write(f"BATCH EVALUATION RESULTS\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Real images base path: {real_base_path}\n")
            f.write(f"Generated images base path: {gen_base_path}\n")
            f.write(f"Styles evaluated: id_{start_id} to id_{end_id}\n")
            f.write(f"Successful evaluations: {successful_evals}/{end_id - start_id + 1}\n")
            f.write(f"Failed evaluations: {failed_evals}\n")
            f.write(f"\n{'=' * 70}\n")
            f.write(f"AVERAGE METRICS (across all styles)\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"  SSIM:  {avg_metrics['SSIM']:.4f} (higher is better)\n")
            f.write(f"  RMSE:  {avg_metrics['RMSE']:.4f} (lower is better)\n")
            f.write(f"  L1:    {avg_metrics['L1']:.4f} (lower is better)\n")
            f.write(f"  LPIPS: {avg_metrics['LPIPS']:.4f} (lower is better)\n")
            f.write(f"  FID:   {avg_metrics['FID']:.4f} (lower is better)\n")
            f.write(f"\n{'=' * 70}\n")
            f.write(f"DETAILED RESULTS BY STYLE\n")
            f.write(f"{'=' * 70}\n")

            for style_id in sorted(all_results.keys()):
                metrics = all_results[style_id]
                f.write(f"\nid_{style_id} ({metrics['num_images']} images):\n")
                f.write(f"  SSIM:  {metrics['SSIM']:.4f}\n")
                f.write(f"  RMSE:  {metrics['RMSE']:.4f}\n")
                f.write(f"  L1:    {metrics['L1']:.4f}\n")
                f.write(f"  LPIPS: {metrics['LPIPS']:.4f}\n")
                f.write(f"  FID:   {metrics['FID']:.4f}\n")

            f.write(f"\n{'=' * 70}\n")

        print(f"Results saved to: {output_file}")
        print(f"{'=' * 70}\n")
    else:
        print(f"\nNo successful evaluations. Please check your folder paths.")


if __name__ == "__main__":
    main()