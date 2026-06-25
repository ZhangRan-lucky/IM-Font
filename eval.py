import os
import re
import glob
import argparse
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision import transforms
from torchvision.models import inception_v3
import lpips

try:
    import cv2
except ImportError:
    raise ImportError("pip install opencv-python")

try:
    import easyocr
except ImportError:
    raise ImportError("pip install easyocr")


def natural_sort_key(path):

    filename = os.path.basename(path)
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', filename)]


def load_char_list(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


_INCEPTION_MODEL = None

def get_inception_model(device='cuda'):
    global _INCEPTION_MODEL
    if _INCEPTION_MODEL is None:
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()
        model = model.to(device)
        model.eval()
        _INCEPTION_MODEL = model
    return _INCEPTION_MODEL


def extract_inception_features(images, device='cuda', batch_size=32):

    model = get_inception_model(device)
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            feat = model(tensors)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def calculate_fid(real_features, gen_features):
    mu_r, sigma_r = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_g, sigma_g = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_g), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean)


_LPIPS_MODEL = None

def get_lpips_model(device='cuda'):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net='alex').to(device)
    return _LPIPS_MODEL


def compute_perceptual_metrics(real_images, generated_images, device='cuda'):
    lpips_metric = get_lpips_model(device)

    ssim_list, rmse_list, l1_list, lpips_list = [], [], [], []

    for real, gen in zip(real_images, generated_images):
        if gen.size != real.size:
            gen = gen.resize(real.size, Image.BICUBIC)

        real_np = np.array(real)
        gen_np = np.array(gen)

        # 修正：自适应 win_size，确保不超过图像尺寸
        min_dim = min(real_np.shape[0], real_np.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        if win_size < 3:
            win_size = 3
            
        ssim_list.append(ssim(real_np, gen_np, channel_axis=2, win_size=win_size, data_range=255))

        real_float = np.array(real).astype(np.float32) / 255.0
        gen_float = np.array(gen).astype(np.float32) / 255.0
        rmse_list.append(np.sqrt(mean_squared_error(real_float.flatten(), gen_float.flatten())))
        l1_list.append(mean_absolute_error(real_float.flatten(), gen_float.flatten()))

        rt = transforms.ToTensor()(real).unsqueeze(0).to(device) * 2 - 1
        gt = transforms.ToTensor()(gen).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            lpips_list.append(lpips_metric(rt, gt).item())

    return {
        'SSIM': float(np.mean(ssim_list)),
        'RMSE': float(np.mean(rmse_list)),
        'L1': float(np.mean(l1_list)),
        'LPIPS': float(np.mean(lpips_list)),
    }


def binarize_glyph(img_pil):
    gray = np.array(img_pil.convert('L'))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(binary > 0) > 0.5:
        binary = 255 - binary
    return binary.astype(np.uint8)


def count_components(binary_img, min_area=5):
    """修正：默认 min_area 从 3 改为 5，过滤噪声"""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            count += 1
    return count


def compute_structural_metrics(real_images, generated_images, min_area=5):
    n_real, n_gen = [], []
    for real, gen in zip(real_images, generated_images):
        if gen.size != real.size:
            gen = gen.resize(real.size, Image.BICUBIC)
        n_real.append(count_components(binarize_glyph(real), min_area))
        n_gen.append(count_components(binarize_glyph(gen), min_area))

    n_real = np.array(n_real, dtype=np.float64)
    n_gen = np.array(n_gen, dtype=np.float64)

    ccr = np.mean(np.minimum(n_real, n_gen) / np.maximum(np.maximum(n_real, n_gen), 1))
    sbr = np.mean(n_gen > n_real)
    mcr = np.mean(n_gen < n_real)

    return {'CCR': float(ccr), 'SBR': float(sbr), 'MCR': float(mcr)}


_OCR_READER = None

def get_ocr_reader(use_gpu=True):
    global _OCR_READER
    if _OCR_READER is None:
        _OCR_READER = easyocr.Reader(['ch_sim'], gpu=use_gpu)
    return _OCR_READER


def ocr_recognize(img_pil, reader, conf_threshold=0.5):


    results = reader.readtext(np.array(img_pil.convert('RGB')), detail=1)
    if results:

        valid_results = [(r[1], r[2]) for r in results if r[2] >= conf_threshold]
        if valid_results:
    
            valid_results.sort(key=lambda x: x[1], reverse=True)
            return valid_results[0][0].strip()
    return ''


def compute_ocr_metrics(real_images, generated_images, reader, gt_chars=None, conf_threshold=0.5):
    correct, total = 0, 0
    for idx, (real, gen) in enumerate(zip(real_images, generated_images)):
        gen_char = ocr_recognize(gen, reader, conf_threshold)
        ref = gt_chars[idx] if (gt_chars is not None and idx < len(gt_chars)) else ocr_recognize(real, reader, conf_threshold)
        if ref == '':
            continue
        total += 1
        correct += (1.0 if gen_char == ref else 0.0)
    return {'OCR': correct / total if total > 0 else float('nan')}


def safe_load_image(path):
    try:
        img = Image.open(path)

        img.load()
        return img.convert('RGB')
    except Exception as e:
        print(f"Warning: Failed to load image {path}: {e}")
        return None


def evaluate_style(style_id, real_base, gen_base, ocr_reader, gt_chars, device, min_area=5):
    real_path = os.path.join(real_base, f'id_{style_id}')
    gen_path = os.path.join(gen_base, f'id_{style_id}')
    if not os.path.exists(real_path) or not os.path.exists(gen_path):
        return None

    real_files = sorted(glob.glob(os.path.join(real_path, '*.png')) + glob.glob(os.path.join(real_path, '*.jpg')),
                        key=natural_sort_key)
    gen_files = sorted(glob.glob(os.path.join(gen_path, '*.png')) + glob.glob(os.path.join(gen_path, '*.jpg')),
                       key=natural_sort_key)
    if not real_files or not gen_files:
        return None

    n = min(len(real_files), len(gen_files))
    

    real_images = []
    gen_images = []
    valid_indices = []
    
    for i in range(n):
        real_img = safe_load_image(real_files[i])
        gen_img = safe_load_image(gen_files[i])
        if real_img is not None and gen_img is not None:
            real_images.append(real_img)
            gen_images.append(gen_img)
            valid_indices.append(i)
    
    if not real_images:
        print(f"Warning: No valid image pairs found for style {style_id}")
        return None

    if gt_chars is not None:
        gt_chars_filtered = [gt_chars[i] for i in valid_indices if i < len(gt_chars)]
    else:
        gt_chars_filtered = None

    perc = compute_perceptual_metrics(real_images, gen_images, device)
    struct = compute_structural_metrics(real_images, gen_images, min_area)
    ocr = compute_ocr_metrics(real_images, gen_images, ocr_reader, gt_chars_filtered)

    real_feat = extract_inception_features(real_images, device)
    gen_feat = extract_inception_features(gen_images, device)
    fid = calculate_fid(real_feat, gen_feat)

    return {**perc, **struct, **ocr, 'FID': fid, 'num_images': len(real_images)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_base_path', required=True)
    parser.add_argument('--gen_base_path', required=True)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=19)
    parser.add_argument('--char_list_file', default=None)
    parser.add_argument('--min_area', type=int, default=5)
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--ocr_conf', type=float, default=0.5, help='OCR confidence threshold')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")

    gt_chars = load_char_list(args.char_list_file)
    ocr_reader = get_ocr_reader(use_gpu=not args.cpu)

    all_metrics = []
    total_images = 0

    for sid in range(args.start_id, args.end_id + 1):
        try:
            res = evaluate_style(sid, args.real_base_path, args.gen_base_path,
                                 ocr_reader, gt_chars, device, args.min_area)
            if res is None:
                continue
            all_metrics.append(res)
            total_images += res['num_images']
            print(f"Style {sid}: {res['num_images']} images evaluated")
        except Exception as e:
            print(f"Error evaluating style {sid}: {e}")
            continue

    if not all_metrics:
        print("No valid images found.")
        return

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0] if k != 'num_images'}

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Styles: {len(all_metrics)}, Total images: {total_images}\n")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

    out = args.output_file or os.path.join(args.gen_base_path, 'eval_full.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Real: {args.real_base_path}\nGen:  {args.gen_base_path}\n")
        f.write(f"Styles: {len(all_metrics)}, Images: {total_images}\n\n")
        for k, v in avg.items():
            f.write(f"{k}: {v:.6f}\n")

    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
