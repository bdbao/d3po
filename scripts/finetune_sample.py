'''
    Input: 
domain [Inpainting/ Outpainting], 
type_polyp [Sessile/ Penduculated],

base_model <hugging face>, 
patched_id <ckpt_id>,
folder_data_images,
folder_data_masks

    Output:
sample.pkl
folder_output_images
'''

import contextlib
import os
import datetime
import time
import sys
import numpy as np
import torch
from functools import partial
import tqdm
from PIL import Image
import json
import pickle
import functools
import inflect
import random
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import PIL
import math
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

# In the config file
config_flags.DEFINE_config_file("config", "config/base_script.py", "Training configuration.")
config = None
# In this sample script
FLAGS = flags.FLAGS
flags.DEFINE_string("domain", "Inpainting", "Specify the domain parameter.")
flags.DEFINE_string("type_polyp", "sessile", "Specify the type_polyp parameter.")
flags.DEFINE_string("base_model", "bdbao/stable-diffusion-inpainting-polyps-nonLoRA-sessile", "Specify the base_model parameter.")
flags.DEFINE_string("patched_id", "", "Specify the checkpoint directory.")
flags.DEFINE_string("folder_data_images", "train_data/kvasir/sessile-polyps/images", "Specify the folder_data_images parameter.")
flags.DEFINE_string("folder_data_masks", "train_data/kvasir/sessile-polyps/masks", "Specify the folder_data_masks parameter.")
flags.DEFINE_string("save_dir", './data', "Specify the save directory.")

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
NUM_PER_PROMPT = 7  # Number of images per prompt
save_dir = None
IE = inflect.engine()

####### utils.py #######
def remove_mid_file(save_dir, num_processes):
    filename = [f'prompt{i}.json' for i in range(num_processes)] + [f'sample{i}.pkl' for i in range(num_processes)] + [f'{i}.txt' for i in range(num_processes)]
    for file in filename:
        try:
            os.remove(os.path.join(save_dir, file))
        except OSError as e:
            print(f"Error: {e}")
def check_data(save_dir, num_processes, sample0_shape):
    with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as f:
        sample: dict = pickle.load(f)
    for key, value in sample.items():
        assert sample0_shape[key][1:]==value.shape[1:] and num_processes*sample0_shape[key][0]==value.shape[0], f'{sample0_shape[key]}{value.shape}'
    remove_mid_file(save_dir, num_processes)
def post_processing(save_dir, num_processes):
    prompts = []
    for i in range(num_processes):
        with open(os.path.join(save_dir, f'prompt{i}.json'), 'r') as f:
            prompts_ = json.load(f)
            prompts += prompts_
    with open(os.path.join(save_dir, 'prompt.json'), 'w') as f:
        json.dump(prompts, f)
    samples = {}
    sample0_shape = {}
    for i in range(num_processes):
        with open(os.path.join(save_dir, f'sample{i}.pkl'), 'rb') as f:
            sample_: dict = pickle.load(f)
            if i==0:
                for key, value in sample_.items():
                    sample0_shape[key] = value.shape
                samples = sample_
            else:
                for key, value in sample_.items():
                    assert sample0_shape[key] == value.shape, f'{key}.shape in sample{i}.pkl({sample0_shape[key]}) is different with {key}.shape in sample0.pkl({value.shape}). '
                samples = {k: torch.cat([s[k] for s in [samples, sample_]]) for k in samples.keys()}
    with open(os.path.join(save_dir, 'sample.pkl'), 'wb') as f:
        pickle.dump(samples, f)
    check_data(save_dir, num_processes, sample0_shape)

####### prompt.py #######
@functools.cache
def _load_images(path, mask_path):
    image_list = []

    def load(p: str):
        pil_image = Image.open(p)

        # Resize the image
        resized_image = pil_image.resize((512, 512))  # Resize to desired dimensions

        return resized_image

    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(path, filename)
            image_mask_path = os.path.join(mask_path, filename)

            image_list.append((load(image_path), load(image_mask_path)))

    return image_list
def from_file(path, low=None, high=None, image: bool = False, mask: str = ""):
    prompts = _load_images(path, mask)[low:high]
    return random.choice(prompts), {}
def kvasir_imgs():
    # return from_file("kvasir/sessile-polyps/images", image = True, mask = "kvasir/sessile-polyps/masks")
    return _load_images(FLAGS.folder_data_images, FLAGS.folder_data_masks) # 20 first images
def kvasir_prompt():
    return 'a photo of polyp', {}

####### ddim_with_logprob.py #######
def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)
def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0, self.alphas_cumprod.gather(0, prev_timestep.cpu()), self.final_alpha_cumprod
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance
def ddim_step_with_logprob(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> Union[DDIMSchedulerOutput, Tuple]:
    
    assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0, self.alphas_cumprod.gather(0, prev_timestep.cpu()), self.final_alpha_cumprod
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob

####### pipeline_with_logprob_inpaint.py #######
def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):
    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image
@torch.no_grad()
def pipeline_with_logprob_inpaint(
    self: StableDiffusionInpaintPipeline,
    prompt: Union[str, List[str]] = None,
    image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    strength: float = 1.0,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
):
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt,
        height,
        width,
        strength,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. set timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(
        num_inference_steps=num_inference_steps, strength=strength, device=device
    )
    # check that number of inference steps is not < 1 - as this doesn't make sense
    if num_inference_steps < 1:
        raise ValueError(
            f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
            f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
        )
    # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
    is_strength_max = strength == 1.0

    # 5. Preprocess mask and image
    mask, masked_image, init_image = prepare_mask_and_masked_image(
        image, mask_image, height, width, return_image=True
    )

    # 6. Prepare latent variables
    num_channels_latents = self.vae.config.latent_channels
    num_channels_unet = self.unet.config.in_channels
    return_image_latents = num_channels_unet == 4

    latents_outputs = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
        image=init_image,
        timestep=latent_timestep,
        is_strength_max=is_strength_max,
        return_noise=True,
        return_image_latents=return_image_latents,
    )

    if return_image_latents:
        latents, noise, image_latents = latents_outputs
    else:
        latents, noise = latents_outputs

    # 7. Prepare mask latent variables
    mask, masked_image_latents = self.prepare_mask_latents(
        mask,
        masked_image,
        batch_size * num_images_per_prompt,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        do_classifier_free_guidance,
    )
    init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
    init_image = self._encode_vae_image(init_image, generator=generator)

    # 8. Check that sizes of mask, masked image and latents match
    if num_channels_unet == 9:
        # default case for runwayml/stable-diffusion-inpainting
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )
    elif num_channels_unet != 4:
        raise ValueError(
            f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
        )

    # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 10. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

    all_latents = [latents]
    all_log_probs = []

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if num_channels_unet == 9:
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents, log_prob = ddim_step_with_logprob(self.scheduler, noise_pred, t, latents, **extra_step_kwargs)
            # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            if num_channels_unet == 4:
                init_latents_proper = image_latents[:1]
                init_mask = mask[:1]

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )

                latents = (1 - init_mask) * init_latents_proper + init_mask * latents

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return image, has_nsfw_concept, all_latents, all_log_probs


def main(argv):
    config = FLAGS.config
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(FLAGS.save_dir, now_time)

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    config_resume_from = FLAGS.patched_id
    if config_resume_from:
        config_resume_from = os.path.normpath(os.path.expanduser(config_resume_from))
        if "checkpoint_" not in os.path.basename(config_resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config_resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config_resume_from}")
            config_resume_from = os.path.join(
                config_resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )

    # set a random seed
    ramdom_seed = np.random.randint(0,100000)
    set_seed(ramdom_seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        FLAGS.base_model, torch_dtype=torch.float16,
    )

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    total_image_num_per_gpu = config.sample.batch_size * config.sample.num_batches_per_epoch * NUM_PER_PROMPT
    global_idx = accelerator.process_index * total_image_num_per_gpu 
    local_idx = 0
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]
        pipeline.unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)
    else:
        trainable_layers = pipeline.unet

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        trainable_layers.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    
    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    if config_resume_from:
        accelerator.load_state(config_resume_from)

    #################### SAMPLING ####################
    pipeline.unet.eval()
    samples = []
    total_prompts = []
    image_idx = 0
    for i in tqdm(
        range(config.sample.num_batches_per_epoch),
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        # generate prompts
        prompts1, prompt_metadata = zip(
            *[kvasir_prompt(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
        )

        masks = []
        input_images = []
        for _ in range(config.sample.batch_size):
            # im, m = kvasir_imgs()[0]
            im, m = kvasir_imgs()[image_idx]
            masks.append(m)
            input_images.append(im)

            image_idx += 1

        # we set the prompts to be the same
        # prompts1 = ["1 hand"] * config.sample.batch_size 
        prompts7 = prompts6 = prompts5 = prompts4 = prompts3 = prompts2 = prompts1
        total_prompts.extend(prompts1)
        # encode prompts
        prompt_ids1 = pipeline.tokenizer(
            prompts1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_ids2 = pipeline.tokenizer(
            prompts2,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_ids3 = pipeline.tokenizer(
            prompts3,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_ids4 = pipeline.tokenizer(
            prompts4,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_ids5 = pipeline.tokenizer(
            prompts5,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_ids6 = pipeline.tokenizer(
            prompts6,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_ids7 = pipeline.tokenizer(
            prompts7,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]
        prompt_embeds2 = pipeline.text_encoder(prompt_ids2)[0]
        prompt_embeds3 = pipeline.text_encoder(prompt_ids3)[0]
        prompt_embeds4 = pipeline.text_encoder(prompt_ids4)[0]
        prompt_embeds5 = pipeline.text_encoder(prompt_ids5)[0]
        prompt_embeds6 = pipeline.text_encoder(prompt_ids6)[0]
        prompt_embeds7 = pipeline.text_encoder(prompt_ids7)[0]
        # sample
        with autocast():
            images1, _, latents1, _ = pipeline_with_logprob_inpaint(
                pipeline,
                image = input_images,
                mask_image = masks,
                prompt_embeds=prompt_embeds1,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
            )
            latents1 = torch.stack(latents1, dim=1)
            images1 = images1.cpu().detach()
            latents1 = latents1.cpu().detach()

            images2, _, latents2, _ = pipeline_with_logprob_inpaint(
                pipeline,
                image = input_images,
                mask_image = masks,
                prompt_embeds=prompt_embeds2,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents2 = torch.stack(latents2, dim=1)
            images2 = images2.cpu().detach()
            latents2 = latents2.cpu().detach()

            images3, _, latents3, _ = pipeline_with_logprob_inpaint(
                pipeline,
                image = input_images,
                mask_image = masks,
                prompt_embeds=prompt_embeds3,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents3 = torch.stack(latents3, dim=1)
            images3 = images3.cpu().detach()
            latents3 = latents3.cpu().detach()

            images4, _, latents4, _ = pipeline_with_logprob_inpaint(
                pipeline,
                image = input_images,
                mask_image = masks,
                prompt_embeds=prompt_embeds4,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents4 = torch.stack(latents4, dim=1)
            images4 = images4.cpu().detach()
            latents4 = latents4.cpu().detach()

            images5, _, latents5, _ = pipeline_with_logprob_inpaint(
                pipeline,
                image = input_images,
                mask_image = masks,
                prompt_embeds=prompt_embeds5,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents5 = torch.stack(latents5, dim=1)
            images5 = images5.cpu().detach()
            latents5 = latents5.cpu().detach()

            images6, _, latents6, _ = pipeline_with_logprob_inpaint(
                pipeline,
                image = input_images,
                mask_image = masks,
                prompt_embeds=prompt_embeds6,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents6 = torch.stack(latents6, dim=1)
            images6 = images6.cpu().detach()
            latents6 = latents6.cpu().detach()
            images7, _, latents7, _ = pipeline_with_logprob_inpaint(
                pipeline,
                image = input_images,
                mask_image = masks,
                prompt_embeds=prompt_embeds7,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents7 = torch.stack(latents7, dim=1)
            images7 = images7.cpu().detach()
            latents7 = latents7.cpu().detach()

        latents = torch.stack([latents1,latents2,latents3,latents4,latents5,latents6,latents7], dim=1)  # (batch_size, 2, num_steps + 1, 4, 64, 64)
        prompt_embeds = torch.stack([prompt_embeds1,prompt_embeds2,prompt_embeds3,prompt_embeds4,prompt_embeds5,prompt_embeds6,prompt_embeds7], dim=1)
        images = torch.stack([images1,images2,images3,images4,images5,images6,images7], dim=1)
        current_latents = latents[:, :, :-1]
        next_latents = latents[:, :, 1:]
        timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)


        samples.append(
            {
                "prompt_embeds": prompt_embeds.cpu().detach(),
                "timesteps": timesteps.cpu().detach(),
                "latents": current_latents.cpu().detach(),  # each entry is the latent before timestep t
                "next_latents": next_latents.cpu().detach(),  # each entry is the latent after timestep t
                "images":images.cpu().detach(),
            }
        )
        os.makedirs(os.path.join(save_dir, "images/"), exist_ok=True)
        if (i+1)%config.sample.save_interval ==0 or i==(config.sample.num_batches_per_epoch-1):
            new_samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            images = new_samples['images'][local_idx:]
            for j, image in enumerate(images):
                for k in range(NUM_PER_PROMPT):
                    pil = Image.fromarray((image[k].cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"), "RGB")
                    pil.save(os.path.join(save_dir, f"images/{(NUM_PER_PROMPT*j+global_idx+k):05}.jpg"))
            global_idx += len(images)*NUM_PER_PROMPT
            local_idx += len(images)
            with open(os.path.join(save_dir, f'prompt{accelerator.process_index}.json'),'w') as f:
                json.dump(total_prompts, f)
            with open(os.path.join(save_dir, f'sample{accelerator.process_index}.pkl'), 'wb') as f:
                pickle.dump({"prompt_embeds": new_samples["prompt_embeds"], "timesteps": new_samples["timesteps"], "latents": new_samples["latents"], "next_latents": new_samples["next_latents"]}, f)
    with open(os.path.join(save_dir, f'{accelerator.process_index}.txt'), 'w') as f:
        f.write(f'{accelerator.process_index} done')
    if accelerator.is_main_process:
        while True:
            done = [True if os.path.exists(os.path.join(save_dir, f'{i}.txt')) else False for i in range(accelerator.num_processes)]
            if all(done):
                time.sleep(5)
                break
        post_processing(save_dir, accelerator.num_processes)

if __name__ == "__main__":
    app.run(main)

# accelerate launch scripts/finetune_sample.py \
# --domain="Inpainting" \
# --type_polyp="sessile" \
# --base_model="bdbao/stable-diffusion-inpainting-polyps-nonLoRA-sessile" \
# --patched_id="logs/using/checkpoints" \
# --folder_data_images="train_data/kvasir/sessile-polyps/images" \
# --folder_data_masks="train_data/kvasir/sessile-polyps/masks" \
# --save_dir="data"