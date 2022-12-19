"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision.transforms as transforms

from matplotlib import pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    colorize_model_and_diffusion_defaults,
    colorize_create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    print(args)
    logger.log("creating model and diffusion...")
    model, diffusion = colorize_create_model_and_diffusion(
        **args_to_dict(args, colorize_model_and_diffusion_defaults().keys())
    )
    #print(model)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu") 
    )

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    #####################################
    logger.log("Loading data ...")
    data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)
    #####################################

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        #model_kwargs = {}
        ############################################
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        ############################################
        #if args.class_cond:
        #    classes = th.randint(
        #        low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        #    )
        #    model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        ##################################################
        ##################################################
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        #if args.class_cond:
        #    gathered_labels = [
        #        th.zeros_like(classes) for _ in range(dist.get_world_size())
        #    ]
        #    dist.all_gather(gathered_labels, classes)
        #    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    temp = arr[0]
    plt.imshow(temp)
    plt.show()
    #if args.class_cond:
    #    label_arr = np.concatenate(all_labels, axis=0)
    #    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            #np.savez(out_path, arr, label_arr)
            np.savez(out_path, arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")

transform = transforms.Compose([transforms.Resize((64, 64))])

def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                print(buffer[0].shape)
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch.permute(0,3,1,2)
                print(batch.shape)
                batch = transform(batch)
                print(batch.shape)
                batch = batch / 127.5 - 1.0
                #batch = batch.permute(0, 3, 1, 2)
                res = dict(gray_scale=batch)
                #res = dict()
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=8,
        use_ddim=False,
        base_samples="",
        model_path="",
    )
    defaults.update(colorize_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
