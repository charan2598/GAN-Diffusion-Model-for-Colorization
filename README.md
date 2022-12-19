# GAN-Diffusion-Model-for-Colorization

This repository contains the code for Image Colorization using Generative Adversarial Network and Guided-Diffusion Model.

## Generative Adversarial Network Model:

The colorization GAN code is available as Jupyter Notebook and is quite straight forward. This GAN model is trained over Stanford Cars, COCO dataset and Landscapes dataset for a fair number of epochs.

Link to GAN checkpoint: [256x256_GAN_checkpoint.pt](https://drive.google.com/file/d/1qgfyvTK-pO4g3QmtEYrJwkfNq7ql6hra/view?usp=share_link)

## Guided-Diffusion Model:

This code is a replica of the OpenAI's Guided-Diffusion code [Original Repo](https://github.com/openai/guided-diffusion), where we have added changes necessary for Image Colorization using Class Conditioned Diffusion.

This model is trained only for a few iterations over just Stanford Cars Dataset.

Link to trained model: [64x64_cond_diffusion.pt](https://drive.google.com/file/d/1QGrDCOLH__M7Xn_REK8X5bDt4a7c8sVR/view?usp=share_link)

You might want to run this command before getting started.
```
pip install -e .
```

### Sampling

The below command can be used to get the images colorized. Here in our code we have trained the diffusion only to enhance the images from GAN model. So in our case the conditioning is over GAN output images and not gray scale images directly.
```
mpiexec -n 1 python scripts/colorize_sample.py --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --noise_schedule linear --num_channels 128 --num_head_channels -1 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --model_path models/64x64_cond_diffusion.pt --base_samples test_image.npz --batch_size 1 --num_samples 1 --timestep_respacing 250 --learn_sigma True
```

For sampling the images need to be stored as a .npz file. Utility code to convert the images of .jpg or .png or any other image formats to .npz format is included in the utility directory.

### Training 

The below command can be used to train the colorization model on a machine with a single GPU of atleast 6GB VRAM.
```
mpiexec -n 1 python scripts/colorize_train.py --data_dir "path/to/orig_dataset" --batch_size 1 --lr 3e-4 --save_interval 100 --log_interval 100 --weight_decay 0.05 --image_size 64 --attention_resolutions 32,16,8 --resblock_updown True --use_scale_shift_norm True --learn_sigma True --num_channels 128 --noise_schedule linear --class_cond True
```

As this is a class conditioned model we will be concatenating the gray scale image or an equivalent to the noise during the sampling phase. So we would need the gray scale or equivalent images in another directory where the original data is present. eg. "path/to/dataset"

## Sample Outputs
### Sample Outputs from GAN

<img src="https://github.com/charan250498/GAN-Diffusion-Model-for-Colorization/blob/master/Images/1.png" width="750">

<img src="https://github.com/charan250498/GAN-Diffusion-Model-for-Colorization/blob/master/Images/2.png" width="750">

### Sample Outputs from Diffusion

<img src="https://github.com/charan250498/GAN-Diffusion-Model-for-Colorization/blob/master/Images/3.png" width="350">
