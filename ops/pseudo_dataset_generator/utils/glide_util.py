"""
Utilities for generating image by glide.
"""

from typing import List, Tuple
import os
from time import time

import torch
import torch.nn.functional as F

from PIL import Image

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

has_cuda = torch.cuda.is_available()
# device = torch.device('cpu' if not has_cuda else 'cuda')

def base_model(device):
    """Create base model."""
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    glide_model, diffusion = create_model_and_diffusion(**options)
    glide_model.eval()
    if has_cuda:
        glide_model.convert_to_fp16()
    glide_model.to(device)
    glide_model.load_state_dict(load_checkpoint('base', device))
    print('total base parameters', sum(x.numel() for x in glide_model.parameters()))
    return glide_model, diffusion, options

def upsampler_model(device):
    """Create upsampler model."""
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
    return model_up, diffusion_up, options_up

def CLIP_model(device):
    """Create CLIP model."""
    clip_model = create_clip_model(device=device)
    clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', device))
    clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', device))
    return clip_model

def save_images(batch: torch.Tensor, tags: List[str] =None, path:str='outputs/', ext:str=".jpg", mode='sep'):
    """ Display a batch of images inline. """
    if not os.path.exists(path):
        os.mkdir(path)
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    if mode == 'all_in_one':
        reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
        imgs = Image.fromarray(reshaped)
        imgs.save(os.path.join(path, "out" + ext), quality=95)
    elif mode == 'sep':
        imgs = [Image.fromarray(arr.permute(1, 2, 0).numpy()) for arr in scaled]
        if tags is None:
            for i, img in enumerate(imgs): 
                img.save(os.path.join(path, str(i) + ext), quality=95)
        elif type(tags) == str:
            for i, img in enumerate(imgs): 
                img.save(os.path.join(path, tags + str(i) + ext), quality=95)
        elif type(tags) == int:
            for i, img in enumerate(imgs): 
                img.save(os.path.join(path, str(tags + i) + ext), quality=95)
        else:
            assert len(tags) == len(imgs)
            for img, tag in zip(imgs, tags):
                img.save(os.path.join(path, tag + ext), quality=95)

def glide(
    prompts: List[str], 
    glide_model, 
    diffusion,
    options,
    clip_model, 
    device,
    guidance_scale = 3.0, 
    path:str='outputs/', 
    ext:str=".jpg", 
    verbose=False):
    """
    Sample from the base model.
    
    Parameters
    ----------------
    prompts. List[str], a list containing the prompts
    """

    batch_size = len(prompts)

    # Create the text tokens to feed to the model.
    tokens = [glide_model.tokenizer.encode(prompt) for prompt in prompts] # List[List[int]]
    outputs = [glide_model.tokenizer.padded_tokens_and_mask(token, options['text_ctx']) for token in tokens]
    tokens, mask = [output[0] for output in outputs], [output[1] for output in outputs]
    # assert type(tokens) == List[List[int]]

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=torch.tensor(tokens, device=device),
        mask=torch.tensor(mask, dtype=torch.bool, device=device)
    )

    # Setup guidance function for CLIP model.
    cond_fn = clip_model.cond_fn(prompts, guidance_scale)

    # Sample from the base model.
    glide_model.del_cache()
    samples = diffusion.p_sample_loop(
        glide_model,
        (batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
    )
    glide_model.del_cache()

    if verbose:
        # Show the output
        save_images(samples, prompts, path=path, ext=ext)
        print("the shape of samples is {}".format(samples.size()))
    
    return samples

def glide_upsampler(
    prompts: List[str], 
    samples:torch.Tensor, 
    glide_model, 
    options,
    model_up, 
    diffusion_up,
    options_up,
    device,
    upsample_temp = 0.997, 
    path:str='outputs/', 
    tags=None,
    ext:str=".jpg", 
    verbose=False):
    """
    Upsample the 64x64 samples.
    """

    batch_size = len(prompts)

    tokens = [glide_model.tokenizer.encode(prompt) for prompt in prompts] # List[List[int]]
    outputs = [glide_model.tokenizer.padded_tokens_and_mask(token, options['text_ctx']) for token in tokens]
    tokens, mask = [output[0] for output in outputs], [output[1] for output in outputs]
    # assert type(tokens) == List[List[int]]

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,

        # Text tokens
        tokens=torch.tensor(
            tokens, device=device
        ),
        mask=torch.tensor(
            mask,
            dtype=torch.bool,
            device=device,
        ),
    )

    # Sample from the base model.
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=torch.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model_up.del_cache()

    if verbose:
        # Show the output
        save_images(up_samples, tags=tags, path=path, ext=ext)
        # print("the shape of up_samples is {}".format(up_samples.size()))

    return up_samples

def generate_single_batch(
    texts: List[str],
    models: Tuple,
    prefix="This is",
    suffix="with white background", 
    path:str='outputs/', 
    tags=None,
    ext:str=".jpg",
    verbose=False,
    quiet=False):
    """
    Create images with respect to texts.
    """
    glide_model, diffusion, options, model_up, diffusion_up, options_up, clip_model, device = models
    glide_config = glide_model, diffusion, options, clip_model, device
    upsampler_config = glide_model, options, model_up, diffusion_up, options_up, device

    prefix = prefix.strip()
    suffix = suffix.strip()
    prompts = [' '.join([prefix, desc.strip(), suffix]) for desc in texts]
    pseudo_images = glide(prompts, *glide_config, path=path, ext=ext, verbose=verbose)
    upsampled_images = glide_upsampler(prompts, pseudo_images, *upsampler_config, path=path, tags=tags, ext=ext, verbose=~quiet)
    upsampled_images = torch.round((upsampled_images + 1) * 255 / 2).clamp(0,255).to(torch.uint8)
    return upsampled_images


def generate_raw_image(
    texts: List[str],
    length: int,
    device:str='cuda:0',
    batch_size: int=25,
    prefix="This is",
    suffix="with white background", 
    path:str='outputs/', 
    ext:str=".jpg"):

    start_iter = int(device[-1]) * length
    device = torch.device(device)

    # Setup
    glide_model, diffusion, options = base_model(device)
    model_up, diffusion_up, options_up = upsampler_model(device)
    clip_model = CLIP_model(device)
    models = glide_model, diffusion, options, model_up, diffusion_up, options_up, clip_model, device

    all_start = time()
    for iter in range(start_iter, start_iter + length):
        start = time()
        slides = texts[iter * batch_size: (iter + 1) * batch_size]
        imgs = generate_single_batch(slides, models, prefix=prefix, tags=iter * batch_size + 1, suffix=suffix, path=path, ext=ext)
        # imgs = F.interpolate(imgs, scale_factor=4, mode='bilinear')
        # save_images(imgs, tags='upsampled_', path=path, ext=ext)
        print("Iter {}/{}: elapsed {}s".format(start_iter + iter + 1, start_iter + length, time() - start), flush=True)
    print("Generation Complete. Total elapsed {}s.".format(time() - all_start), flush=True)
