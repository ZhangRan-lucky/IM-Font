# IM-Font

Official implementation of **IM-Font: One-Shot Chinese Font Generation via IDS Structural Encoding and Multi-Stage Style Injection**.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19422224.svg)](https://doi.org/10.5281/zenodo.19422224)

## Dependencies

```bash
pip install -r requirements.txt
```


## Dataset


[方正字库](https://www.foundertype.com/index.php/FindFont/index) provides free font download for non-commercial users.

Example directory hierarchy

    data_dir
        |--- id_1
        |--- id_2
               |--- 00001.png
               |--- 00002.png
               |--- ...
        |--- ...




## Usage

------

### Prepare dataset

```python
python font2img.py --ttf_path ttf_folder --chara total_chn.txt --save_path save_folder --img_size 80 --chara_size 60
```
### Training
```python
python train.py --cfg_path cfg/train_cfg.yaml
```
### Testing
```python
python sample.py --cfg_path cfg/test_cfg.yaml --style_img style.png --char "好"
```


# Acknowledgements

------

This project is based on [Diff-Font](https://github.com/Hxyz-123/Font-diff) and [openai/guided-diffusion](https://github.com/openai/guided-diffusion). The style encoder is adapted from [DG-Font](https://github.com/ecnuycxie/DG-Font)
