import os
import argparse
from utils import dist_util, logger
from utils.image_datasets import load_data
from utils.resample import create_named_schedule_sampler
from utils.script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    create_model_and_diffusion,
)
from utils.train_util import TrainLoop
import torch as th
from attrdict import AttrDict
import yaml

from pathlib import Path

try:
    import cn
    cn.RAW_FILE_DIR = Path('/path/to/your/data/')
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./cfg/train_cfg.yaml',
                        help='config file path')
    parser = parser.parse_args()
    with open(parser.cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = AttrDict(create_cfg(cfg))

    train_step = cfg.train_step
    total_train_step = cfg.total_train_step
    sty_encoder_path = cfg.sty_encoder_path
    classifier_free = cfg.classifier_free
    use_ids = cfg.get('use_ids', True)
    ids_max_len = cfg.get('ids_max_len', 40)
    ids_vocab_size = cfg.get('ids_vocab_size')
    num_stages = cfg.get('num_stages', 3)

    cfg.__delattr__('train_step')
    cfg.__delattr__('total_train_step')
    cfg.__delattr__('sty_encoder_path')
    cfg.__delattr__('classifier_free')

    dist_util.setup_dist()

    model_save_dir = cfg.model_save_dir

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    logger.configure(dir=model_save_dir, format_strs=['stdout', 'log', 'csv'])

    logger.log("=" * 60)
    logger.log("=" * 60)

    data = load_data(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        classifier_free=classifier_free,
        total_char_file=cfg.total_char_file,
        use_ids=use_ids,
        ids_max_len=ids_max_len,
    )

    if use_ids:
        if ids_vocab_size is None:
            raise ValueError(
            )


    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cfg, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(cfg.schedule_sampler, diffusion)

    if not cfg.resume_checkpoint:
        logger.log("=" * 60)
        checkpoint = th.load(sty_encoder_path, map_location='cpu')

        tmp_dict = {}
        for k, v in checkpoint.items():
            if k in model.sty_encoder.state_dict():
                tmp_dict[k] = v

        if tmp_dict:
            model.sty_encoder.load_state_dict(tmp_dict)


        if classifier_free:
            raise ValueError(
                f"required conditional trained model, please fill in the model path in 'resume_checkpoint'")
    else:
        logger.log("=" * 60)
        logger.log(f"checkpoint: {cfg.resume_checkpoint}")
        logger.log("=" * 60)

    # frozen sty_encoder
    for p in model.sty_encoder.parameters():
        p.requires_grad = False

    logger.log("\n" + "=" * 80)
    logger.log("Parameter Freeze Status Check:")
    logger.log("=" * 80)

    sty_encoder_trainable = sum(p.numel() for p in model.sty_encoder.parameters() if p.requires_grad)
    sty_encoder_total = sum(p.numel() for p in model.sty_encoder.parameters())
    logger.log(f"sty_encoder: {sty_encoder_trainable}/{sty_encoder_total} trainable (should be 0)")

    stage_base_trainable = sum(p.numel() for p in model.stage_base_proj.parameters() if p.requires_grad)
    stage_base_total = sum(p.numel() for p in model.stage_base_proj.parameters())
    logger.log(f"stage_base_proj: {stage_base_trainable}/{stage_base_total} trainable")

    stage_mod_trainable = sum(p.numel() for p in model.stage_modulators.parameters() if p.requires_grad)
    stage_mod_total = sum(p.numel() for p in model.stage_modulators.parameters())
    logger.log(f"stage_modulators: {stage_mod_trainable}/{stage_mod_total} trainable")

    if use_ids:
        ids_encoder_trainable = sum(p.numel() for p in model.ids_encoder.parameters() if p.requires_grad)
        ids_encoder_total = sum(p.numel() for p in model.ids_encoder.parameters())
        logger.log(f"ids_encoder: {ids_encoder_trainable}/{ids_encoder_total} trainable")

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Total: {total_trainable}/{total_params} trainable parameters")
    logger.log(f"Number of stages: {model.num_stages}")

    logger.log(f"\n🔥 Learnable Weights:")
    logger.log(f"  stage_weight: {model.stage_weight.item():.4f}")
    if use_ids:
        logger.log(f"  ids_weight: {model.ids_weight.item():.4f}")

    assert sty_encoder_trainable == 0, "sty_encoder should be completely frozen!"


    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=cfg.batch_size,
        microbatch=cfg.microbatch,
        lr=cfg.lr,
        ema_rate=cfg.ema_rate,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        train_step=train_step,
        resume_checkpoint=cfg.resume_checkpoint,
        use_fp16=cfg.use_fp16,
        fp16_scale_growth=cfg.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.weight_decay,
        classifier_free=classifier_free,
        total_train_step=total_train_step
    ).run_loop()


def create_cfg(cfg):
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=250,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        attention_resolutions='40, 20, 10',
        total_char_file=None,
        use_ids=True,
        ids_max_len=40,
        ids_vocab_size=None,
        num_stages=3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cfg)
    return defaults


if __name__ == "__main__":
    main()