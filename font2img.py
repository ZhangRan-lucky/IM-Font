from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import pathlib
from fontTools.ttLib import TTFont, TTCollection
import argparse
import hashlib
import random

parser = argparse.ArgumentParser(description='Generate character images from .ttf/.ttc')
parser.add_argument('--ttf_dir', type=str, required=True, help='Directory of .ttf/.ttc fonts')
parser.add_argument('--chara', type=str, required=True, help='Character file path')
parser.add_argument('--save_root', type=str, required=True, help='Root directory to save images')
parser.add_argument('--start_font_idx', type=int, default=0, help='Starting font index')
parser.add_argument('--num_fonts', type=int, default=20, help='Number of fonts to generate')
parser.add_argument('--img_size', type=int, default=80, help='Canvas size of generated images')
parser.add_argument('--chara_size', type=int, default=64, help='Font size for characters')
parser.add_argument('--num_chars', type=int, default=4000, help='Number of characters to generate')
parser.add_argument('--failed_log', type=str, default='failed_fonts.txt', help='File to log failed fonts')
parser.add_argument('--check_chars', type=int, default=20, help='Number of characters to check for damage')
parser.add_argument('--min_fill', type=float, default=0.02, help='Minimum fill ratio for a valid font')
parser.add_argument('--max_fill', type=float, default=0.9, help='Maximum fill ratio for a valid font')
args = parser.parse_args()

with open(args.chara, 'r', encoding='utf-8') as f:
    all_chars = f.read().strip()
    characters = all_chars[:args.num_chars]
print(f"Loaded {len(characters)} characters from {args.chara}")


def draw_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img


def compute_fill_ratio(img):
    arr = np.array(img)
    black_pixels = np.sum(arr < 200)
    total_pixels = arr.size
    return black_pixels / total_pixels


def check_font_quality(font_path, pil_font, sample_chars):
    fill_ratios = []
    for ch in sample_chars:
        img = draw_char(ch, pil_font, args.img_size,
                        (args.img_size - args.chara_size) / 2,
                        (args.img_size - args.chara_size) / 2)
        ratio = compute_fill_ratio(img)
        fill_ratios.append(ratio)
    avg_fill = np.mean(fill_ratios)
    if avg_fill < args.min_fill or avg_fill > args.max_fill:
        return False, avg_fill
    return True, avg_fill


def check_font_complete(font_path, characters, font_index=0):
    try:
        if font_path.lower().endswith('.ttc'):
            ttc = TTCollection(font_path)
            font_obj = ttc.fonts[font_index]
        else:
            font_obj = TTFont(font_path)
        cmap = font_obj.getBestCmap()
        font_chars = set(chr(u) for u in cmap.keys())
        missing = [c for c in characters if c not in font_chars]
        if missing:
            return False, f"Missing {len(missing)} characters"
        return True, ""
    except Exception as e:
        return False, str(e)


def get_file_hash(font_path):
    hasher = hashlib.md5()
    with open(font_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def generate_images(font_paths, save_dir, start_idx=0):
    os.makedirs(save_dir, exist_ok=True)
    failed_fonts = []
    seen_hashes = set()
    seen_names = set()

    for idx, font_path in enumerate(font_paths):
        try:
            font_hash = get_file_hash(font_path)
            font_name = os.path.basename(font_path).split('.')[0]
            if font_hash in seen_hashes or font_name in seen_names:
                print(f"Skipping duplicate font: {font_name}")
                continue
            seen_hashes.add(font_hash)
            seen_names.add(font_name)

            if font_path.lower().endswith('.ttc'):
                ttc = TTCollection(font_path)
                for i, font_obj in enumerate(ttc.fonts):
                    pil_font = ImageFont.truetype(font_path, size=args.chara_size, index=i)
                    sample_chars = random.sample(characters, min(args.check_chars, len(characters)))
                    ok, fill = check_font_quality(font_path, pil_font, sample_chars)
                    if not ok:
                        print(f"Skipping damaged font: {font_name} idx{i} (avg_fill={fill:.3f})")
                        failed_fonts.append((font_path, f"Low-quality font (avg_fill={fill:.3f})"))
                        continue

                    font_save_dir = os.path.join(save_dir, f'id_{start_idx + idx}_idx{i}')
                    os.makedirs(font_save_dir, exist_ok=True)
                    for char_idx, char in enumerate(characters):
                        img = draw_char(char, pil_font, args.img_size,
                                        (args.img_size - args.chara_size) / 2,
                                        (args.img_size - args.chara_size) / 2)
                        img.save(os.path.join(font_save_dir, f'{char_idx + 1:05d}.png'))
                    print(f"Font {start_idx + idx}: OK (avg_fill={fill:.3f}) {font_name} idx{i}")

            else:
                pil_font = ImageFont.truetype(font_path, size=args.chara_size)
                sample_chars = random.sample(characters, min(args.check_chars, len(characters)))
                ok, fill = check_font_quality(font_path, pil_font, sample_chars)
                if not ok:
                    print(f"Skipping damaged font: {font_name} (avg_fill={fill:.3f})")
                    failed_fonts.append((font_path, f"Low-quality font (avg_fill={fill:.3f})"))
                    continue

                font_save_dir = os.path.join(save_dir, f'id_{start_idx + idx}')
                os.makedirs(font_save_dir, exist_ok=True)
                for char_idx, char in enumerate(characters):
                    img = draw_char(char, pil_font, args.img_size,
                                    (args.img_size - args.chara_size) / 2,
                                    (args.img_size - args.chara_size) / 2)
                    img.save(os.path.join(font_save_dir, f'{char_idx + 1:05d}.png'))
                print(f"Font {start_idx + idx}: OK (avg_fill={fill:.3f}) {font_name}")

        except Exception as e:
            failed_fonts.append((font_path, str(e)))
            print(f"Failed to process font {font_path}: {e}")
    return failed_fonts


def get_complete_fonts(ttf_dir, characters):
    ttf_paths = list(pathlib.Path(ttf_dir).glob('*.*'))
    complete_fonts = []
    failed_fonts = []
    seen_hashes = set()

    for path in ttf_paths:
        try:
            fhash = get_file_hash(str(path))
            if fhash in seen_hashes:
                print(f"Skipping duplicate file: {path}")
                continue
            seen_hashes.add(fhash)
            success, reason = check_font_complete(str(path), characters)
            if success:
                complete_fonts.append(str(path))
            else:
                failed_fonts.append((str(path), reason))
                print(f"Skipping {path}: {reason}")
        except Exception as e:
            failed_fonts.append((str(path), str(e)))
            print(f"Error reading {path}: {e}")

    print(f"Found {len(complete_fonts)} complete fonts in {ttf_dir}")
    return complete_fonts, failed_fonts


all_fonts, all_failed = get_complete_fonts(args.ttf_dir, characters)

end_idx = args.start_font_idx + args.num_fonts
if len(all_fonts) < end_idx:
    print(f"Warning: Only {len(all_fonts)} fonts available, but need fonts from index {args.start_font_idx} to {end_idx-1}")
    selected_fonts = all_fonts[args.start_font_idx:]
else:
    selected_fonts = all_fonts[args.start_font_idx:end_idx]

print(f"\n=== Processing fonts from index {args.start_font_idx} to {args.start_font_idx + len(selected_fonts) - 1} ===")
print(f"Total fonts to process: {len(selected_fonts)}")

failed_fonts = generate_images(selected_fonts, os.path.join(args.save_root, 'train_images'), start_idx=args.start_font_idx)

all_failed_list = all_failed + failed_fonts
if all_failed_list:
    print("\n==== Failed Fonts ====")
    with open(args.failed_log, 'w', encoding='utf-8') as f:
        for fpath, reason in all_failed_list:
            print(f"{fpath} -> {reason}")
            f.write(f"{fpath} -> {reason}\n")

print(f"\n=== Summary ===")
print(f"Fonts processed: {len(selected_fonts)}")
print(f"Failed fonts: {len(all_failed_list)}")
print(f"Failed fonts list saved to {args.failed_log}")
