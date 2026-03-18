import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from utils import dist_util, logger
from utils.script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    create_model_and_diffusion,
)
from PIL import Image
from attrdict import AttrDict
import yaml

from pathlib import Path

try:
    import cn
    cn.RAW_FILE_DIR = Path('/path/to/your/data/')
    CN_AVAILABLE = True
except ImportError:
    CN_AVAILABLE = False


def img_pre_pros(img_path, image_size):
    pil_image = Image.open(img_path).resize((image_size, image_size))
    pil_image.load()
    pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    arr = arr.astype(np.float32) / 127.5 - 1
    return np.transpose(arr, [2, 0, 1])


def load_ids_data(total_char_file, expected_vocab_size, ids_max_len=40):

    with open(total_char_file, 'r', encoding='utf-8') as f:
        total_chars = f.read().strip()

    logger.log(f"总字符数: {len(total_chars)}")

    try:
        ids_map, ids_set, ids_base = cn.resolve_IDS_babelstone(
            total_chars,
            level='stroke'
        )

    special_tokens = ['<PAD>', '<UNK>']
    base_components = sorted(ids_base.keys())
    ids_vocab = special_tokens + list(cn.IDC) + base_components
    ids2idx = {v: i for i, v in enumerate(ids_vocab)}

    actual_vocab_size = len(ids_vocab)

    char2ids = {}
    failed_chars = []
    ids_lengths = []

    for char_idx, char in enumerate(total_chars):
        ids_tuple = ids_map.get(char, (char,))


        if ids_tuple == (char,) and char not in ids_base:
            failed_chars.append(char)

        ids_indices = [ids2idx.get(c, ids2idx['<UNK>']) for c in ids_tuple]
        ids_lengths.append(len(ids_indices))

        # Padding到固定长度
        if len(ids_indices) < ids_max_len:
            ids_indices += [ids2idx['<PAD>']] * (ids_max_len - len(ids_indices))
        else:
            ids_indices = ids_indices[:ids_max_len]

        char2ids[char_idx] = np.array(ids_indices, dtype=np.int64)


    logger.log("=" * 50)

    return char2ids, ids2idx, actual_vocab_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./cfg/test_cfg.yaml',
                        help='config file path')
    parser = parser.parse_args()
    with open(parser.cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = AttrDict(create_cfg(cfg))

    model_path = cfg.model_path
    sty_img_path = cfg.sty_img_path
    total_txt_file = cfg.total_txt_file
    gen_txt_file = cfg.gen_txt_file
    img_save_path = cfg.img_save_path
    classifier_free = cfg.classifier_free
    cont_gudiance_scale = cfg.cont_scale
    sk_gudiance_scale = cfg.sk_scale
    use_ids = cfg.get('use_ids', False)
    ids_max_len = cfg.get('ids_max_len', 40)
    expected_vocab_size = cfg.get('ids_vocab_size', None)

    cfg.__delattr__('model_path')
    cfg.__delattr__('sty_img_path')
    cfg.__delattr__('total_txt_file')
    cfg.__delattr__('gen_txt_file')
    cfg.__delattr__('img_save_path')
    cfg.__delattr__('classifier_free')
    cfg.__delattr__('cont_scale')
    cfg.__delattr__('sk_scale')

    dist_util.setup_dist()

    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)


        try:
            char2ids, ids2idx, actual_vocab_size = load_ids_data(
                total_txt_file,
                expected_vocab_size,
                ids_max_len
            )


    logger.log("=" * 60)
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cfg, model_and_diffusion_defaults().keys())
    )


    model.to(dist_util.dev())
    if cfg.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # 检查风格图片是否存在
    if not os.path.exists(sty_img_path):
        logger.log(f"Error: Style image not found: {sty_img_path}")
        return

    logger.log(f"\n{'=' * 60}")
    logger.log(f"Style image: {sty_img_path}")
    logger.log(f"Save path: {img_save_path}")
    logger.log(f"{'=' * 60}")

    char2idx = {}
    with open(total_txt_file, 'r', encoding='utf-8') as f:
        chars = f.read().strip()
        for idx, char in enumerate(chars):
            char2idx[char] = idx

    char_idx = []
    with open(gen_txt_file, 'r', encoding='utf-8') as f1:
        genchars = f1.read().strip()
        for char in genchars:
            if char in char2idx:
                char_idx.append(char2idx[char])


    logger.log(f"Will generate {len(char_idx)} characters")
    logger.log("sampling...")

    all_images = []
    all_labels = []
    ch_idx = 0

    while len(all_images) * cfg.batch_size < cfg.num_samples:
        model_kwargs = {}
        remaining = cfg.num_samples - len(all_images) * cfg.batch_size
        current_batch_size = min(cfg.batch_size, remaining)


        classes = th.tensor([i for i in char_idx[ch_idx:ch_idx + current_batch_size]],
                            device=dist_util.dev())
        ch_idx += current_batch_size

        model_kwargs["y"] = classes
        img = th.tensor(img_pre_pros(sty_img_path, cfg.image_size),
                        requires_grad=False).cuda().repeat(current_batch_size, 1, 1, 1)
        model_kwargs["sty"] = img

        if use_ids and char2ids is not None:
            ids_batch = []
            for cls_idx in classes.cpu().numpy():
                ids_batch.append(char2ids[cls_idx])
            model_kwargs["ids"] = th.tensor(np.stack(ids_batch),
                                            dtype=th.int64).to(dist_util.dev())

        # ===== Classifier-Free Guidance =====
        if classifier_free:
            if use_ids and char2ids is not None:
                model_kwargs["mask_y"] = th.cat([
                    th.zeros([current_batch_size], dtype=th.bool),
                    th.ones([current_batch_size * 2], dtype=th.bool)
                ]).to(dist_util.dev())
                model_kwargs["y"] = model_kwargs["y"].repeat(3)

                model_kwargs["mask_ids"] = th.cat([
                    th.ones([current_batch_size], dtype=th.bool),
                    th.zeros([current_batch_size], dtype=th.bool),
                    th.ones([current_batch_size], dtype=th.bool)
                ]).to(dist_util.dev())
                model_kwargs["ids"] = model_kwargs["ids"].repeat(3, 1)
                model_kwargs["sty"] = model_kwargs["sty"].repeat(3, 1, 1, 1)
            else:
                # 不使用IDS的CFG
                model_kwargs["mask_y"] = th.cat([
                    th.zeros([current_batch_size], dtype=th.bool),
                    th.ones([current_batch_size], dtype=th.bool)
                ]).to(dist_util.dev())
                model_kwargs["y"] = model_kwargs["y"].repeat(2)
                model_kwargs["sty"] = model_kwargs["sty"].repeat(2, 1, 1, 1)
        else:
            model_kwargs["mask_y"] = th.zeros([current_batch_size], dtype=th.bool).to(dist_util.dev())
            if use_ids and char2ids is not None:
                model_kwargs["mask_ids"] = th.zeros([current_batch_size], dtype=th.bool).to(dist_util.dev())

        def model_fn(x_t, ts, **model_kwargs):
            if classifier_free:
                repeat_time = model_kwargs["y"].shape[0] // x_t.shape[0]
                x_t = x_t.repeat(repeat_time, 1, 1, 1)
                ts = ts.repeat(repeat_time)

                if use_ids and char2ids is not None:
                    model_output = model(x_t, ts, **model_kwargs)
                    model_output_y, model_output_ids, model_output_uncond = model_output.chunk(3)
                    model_output = model_output_uncond + \
                                   cont_gudiance_scale * (model_output_y - model_output_uncond) + \
                                   sk_gudiance_scale * (model_output_ids - model_output_uncond)
                else:
                    model_output = model(x_t, ts, **model_kwargs)
                    model_output_cond, model_output_uncond = model_output.chunk(2)
                    model_output = model_output_uncond + cont_gudiance_scale * (
                            model_output_cond - model_output_uncond)
            else:
                model_output = model(x_t, ts, **model_kwargs)
            return model_output

        sample_fn = (
            diffusion.p_sample_loop if not cfg.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (current_batch_size, 3, cfg.image_size, cfg.image_size),
            clip_denoised=cfg.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_labels = [
            th.zeros_like(classes) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        actual_generated = min(len(all_images) * current_batch_size, cfg.num_samples)
        logger.log(f"created {actual_generated} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: cfg.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: cfg.num_samples]

    if dist.get_rank() == 0:
        for idx, (img_sample, img_cls) in enumerate(zip(arr, label_arr)):
            img = Image.fromarray(img_sample).convert("RGB")
            img_name = "%05d.png" % (idx + 1)
            img.save(os.path.join(img_save_path, img_name))

    logger.log(f"\n{'=' * 60}")
    logger.log("Sampling complete!")
    logger.log(f"Saved {len(arr)} images to {img_save_path}")
    logger.log(f"{'=' * 60}")

    dist.barrier()


def create_cfg(cfg):
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=16,
        use_ddim=False,
        model_path="",
        cont_scale=1.0,
        sk_scale=1.0,
        sty_img_path="",
        use_ids=False,
        ids_vocab_size=None,
        ids_max_len=40,
        attention_resolutions='40, 20, 10',
        total_txt_file=None,
        gen_txt_file=None,
        img_save_path='./result',
        classifier_free=False,
        num_stages=3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cfg)
    return defaults


if __name__ == "__main__":
    main()