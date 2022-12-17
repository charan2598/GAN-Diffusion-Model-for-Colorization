"""
Train a colorization model on images.
"""

import argparse

import torch
from torchvision.transforms import transforms
from PIL import Image
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    colorize_model_and_diffusion_defaults,
    colorize_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    print(args)
    logger.log("creating colorization model and diffusion...")
    model, diffusion = colorize_create_model_and_diffusion(
        **args_to_dict(args, colorize_model_and_diffusion_defaults().keys())
    )
    #print(model)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_colorize_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

###################################################################

###################################################################

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

def load_colorize_data(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir = data_dir,
        batch_size = batch_size,
        image_size = image_size,
        class_cond = class_cond,
    )
    for color_batch, model_kwargs in data:
        cond_list = []
        for i, path in enumerate(model_kwargs["path"]):
            model_kwargs["path"][i] = model_kwargs["path"][i].replace("orig_land", "land")
            cond_list.append(transform(Image.open(model_kwargs["path"][i])).unsqueeze(0))
        model_kwargs.pop("path")
        model_kwargs["gray_scale"] = torch.cat(tuple(cond_list), dim=0)
        yield color_batch, model_kwargs

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(colorize_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
