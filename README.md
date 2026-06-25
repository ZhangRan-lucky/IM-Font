# IM-Font

Official implementation of **IM-Font: IDS-Guided Multi-Stage Diffusion for One-Shot Chinese Font Generation**.
This repository contains the official implementation of our paper, including training, evaluation, and inference scripts for one-shot Chinese font generation.
This release is archived on Zenodo: [https://doi.org/10.5281/zenodo.19422461](https://zenodo.org/records/19422461)




## Dataset

We use fonts from [Founder Type](https://www.foundertype.com/) for training and evaluation. 
Due to licensing restrictions, we cannot redistribute the original font files (.ttf).

To prepare the dataset:
1. Download fonts from the Founder Type website (free for non-commercial use).
2. Convert the font files to PNG images using our provided script:

The dataset is split into training and test sets:

```bash
# Training set: font indices 0 ~ 80 (80 fonts)
python font2img.py --ttf_dir /path/to/fonts --chara total_chn.txt --save_root ./datastes --start_font_idx 0 --num_fonts 80

# Test set: font indices 80 ~ 100 (20 fonts)
python font2img.py --ttf_dir /path/to/fonts --chara total_chn.txt --save_root ./datastes --start_font_idx 100 --num_fonts 20

Example directory hierarchy

    datasets/
    ├── train_images/
    │   ├── id_0/
    │   ├── id_1/
    │   └── ... (id_0 ~ id_70)
    └── test_images/
        ├── id_80/
        ├── id_99/
        └── ... (id_80 ~ id_99)

The required data files are provided in the repository:

- `total_chn.txt`  : Complete character set for training and evaluation.
- `gen_char.txt`   : Character set for inference / generation.
- `data/raw_files/`     : BabelStone IDS source data for the `cn` module.


## Usage

------

### Environment Setup
1. Create a conda environment with Python 3.10.14 and activate it:
    ```bash
    conda create -n imfont python=3.10.14 -y
    conda activate imfont
2. Install PyTorch 2.2.2 with CUDA 11.8:
```python
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```
3. Install other dependencies from the provided requirements.txt file in the root directory:
```python
pip install -r requirements.txt
```

### Train source model

You have two options:
1. Our pre-trained models are available at [Google Drive](https://drive.google.com/drive/folders/1eaYxbhV9ipaJJoXVHilryIdUZzUV01Fg?usp=sharing).
Place the downloaded `.ckpt` and `.pt` files under `./pretrained_models/` before running the code.
2. Train your own models

To train your own models, please run the following commands.
```python
python train.py --cfg_path cfg/train_cfg.yaml
```
### Testing
```python
python sample.py --cfg_path cfg/test_cfg.yaml --style_img style.png --char "好"
```

# Acknowledgements

------

This project is built upon the following excellent open-source works:
- The diffusion framework is based on [Diff-Font](https://github.com/Hxyz-123/Font-diff) and [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
- The style encoder is adapted from [DG-Font](https://github.com/ecnuycxie/DG-Font).
- The IDS sequence dataset and processing pipeline are adapted from [IF-Font](https://github.com/Stareven233/IF-Font) (NeurIPS 2024).
