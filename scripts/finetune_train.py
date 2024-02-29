'''
    Input: 
domain [Inpainting/ Outpainting],
type [Sessile/ Penduculated],

base_model <hugging face>,
patched_id <ckpt_id>,
sample.pkl
feedbacks.json

    Output:
new_patched_id <ckpt_id>
'''

import contextlib
import os
import copy
import datetime
import sys
from concurrent import futures
import torch
from functools import partial
import tqdm
import tree
from itertools import combinations
import numpy as np
import json
import pickle
from typing import Optional, Tuple, Union
import math
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
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
flags.DEFINE_string("sample_pkl_path", "data/using", "Specify the samples location parameter.")
flags.DEFINE_string("feedbacks_json_path", "data/using/json", "Specify the feedbacks directory.")
flags.DEFINE_string("save_dir", 'logs', "Specify the save directory.")


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

####### utils.py #######
def load_data_from_json(path):
    json_files = [f for f in os.listdir(path) if f.endswith('.json')]
    all_data = []
    for file in json_files:
        with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)
            all_data.append(data)
    min_length = min(map(len, all_data))
    data_clip = np.array([l[:min_length] for l in all_data])
    all_data_np = np.array(data_clip)
    mean_values = np.mean(all_data_np, axis=0)
    return mean_values
def load_sample(path):
    with open(os.path.join(path,"sample.pkl"),'rb') as f:
        sample = pickle.load(f)
    return sample

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

def main(argv):
    config = FLAGS.config

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
        project_dir=os.path.join(FLAGS.patched_id, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="d3po-pytorch", config=config.to_dict(), init_kwargs={"wandb": {"name": config.run_name}}
        )

    ramdom_seed = np.random.randint(0,100000)
    set_seed(ramdom_seed, device_specific=True)

    # load scheduler, tokenizer and models.
    ### pipeline = StableDiffusionInpaintPipeline.from_pretrained(FLAGS.base_model, torch_dtype=torch.float16)
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16)
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
        pipeline.unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)
    else:
        trainable_layers = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                ### FLAGS.base_model, revision=config.pretrained.revision_inpaint, subfolder="unet"
                config.pretrained.model, revision=config.pretrained.revision_inpaint, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
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
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config_resume_from:
        accelerator.load_state(config_resume_from)
        
    ref =  copy.deepcopy(pipeline.unet)
    for param in ref.parameters():
        param.requires_grad = False
    # get sample dict
    samples = load_sample(FLAGS.sample_pkl_path)
    # get human preference
    human_prefer = load_data_from_json(FLAGS.feedbacks_json_path)
    human_prefer = torch.tensor(human_prefer)
    samples["human_prefer"] = human_prefer
    samples = tree.map_structure(lambda x: x if isinstance(x, torch.Tensor) else x, samples)
    pipeline.scheduler.timesteps = samples['timesteps']
    pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
    combinations_list = list(combinations(range(7), 2))
    num_timesteps = samples["timesteps"].shape[1]
    total_batch_size = human_prefer.shape[0]
    assert num_timesteps == config.sample.num_steps
    init_samples = copy.deepcopy(samples)
    for epoch in range(config.num_epochs):
        # shuffle samples along batch dimension
        perm = torch.randperm(total_batch_size)
        samples = {k: v[perm] for k, v in init_samples.items()}

        # shuffle along time dimension independently for each sample
        perms = torch.stack(
            [torch.randperm(num_timesteps) for _ in range(total_batch_size)]
        )
        for key in ["latents", "next_latents"]:
            tmp = samples[key].permute(0,2,3,4,5,1)[torch.arange(total_batch_size)[:, None], perms]
            samples[key] = tmp.permute(0,5,1,2,3,4)
            del tmp
        samples["timesteps"] = samples["timesteps"][torch.arange(total_batch_size)[:, None], perms].unsqueeze(1).repeat(1,7,1)
        # training
        pipeline.unet.train()
        for i in tqdm(range(0,total_batch_size,config.train.batch_size),
                    desc="Update",
                    position=2,
                    leave=False, 
                        ):
            if ((i+1) // config.train.batch_size) % config.train.save_interval==0:
                accelerator.save_state()
            for each_combination in combinations_list:
                sample_0 = tree.map_structure(lambda value: value[i:i+config.train.batch_size, each_combination[0]].to(accelerator.device), samples)
                sample_1 = tree.map_structure(lambda value: value[i:i+config.train.batch_size, each_combination[1]].to(accelerator.device), samples)
                if torch.all(sample_0['human_prefer'] == sample_1['human_prefer']): 
                    continue
                # compute which image is better
                compare_sample0 = (sample_0['human_prefer'] > sample_1['human_prefer']).int() * 2 - 1
                compare_sample1 = (sample_1['human_prefer'] > sample_0['human_prefer']).int() * 2 - 1
                equal_mask = sample_0['human_prefer'] == sample_1['human_prefer']
                compare_sample0[equal_mask] = 0
                compare_sample1[equal_mask] = 0
                human_prefer = torch.stack([compare_sample0, compare_sample1], dim=1)

                
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds_0 = torch.cat([train_neg_prompt_embeds, sample_0["prompt_embeds"]])
                    embeds_1 = torch.cat([train_neg_prompt_embeds, sample_1["prompt_embeds"]])
                else:
                    embeds_0 = sample_0["prompt_embeds"]
                    embeds_1 = sample_1["prompt_embeds"]
                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=3,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):  
                    with accelerator.accumulate(pipeline.unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred_0 = pipeline.unet(
                                    torch.cat([sample_0["latents"][:, j]] * 2),
                                    torch.cat([sample_0["timesteps"][:, j]] * 2),
                                    embeds_0,
                                ).sample
                                noise_pred_uncond_0, noise_pred_text_0 = noise_pred_0.chunk(2)
                                noise_pred_0 = noise_pred_uncond_0 + config.sample.guidance_scale * (noise_pred_text_0 - noise_pred_uncond_0)

                                noise_ref_pred_0 = ref(
                                    torch.cat([sample_0["latents"][:, j]] * 2),
                                    torch.cat([sample_0["timesteps"][:, j]] * 2),
                                    embeds_0,
                                ).sample
                                noise_ref_pred_uncond_0, noise_ref_pred_text_0 = noise_ref_pred_0.chunk(2)
                                noise_ref_pred_0 = noise_ref_pred_uncond_0 + config.sample.guidance_scale * (
                                    noise_ref_pred_text_0 - noise_ref_pred_uncond_0
                                )

                                noise_pred_1 = pipeline.unet(
                                    torch.cat([sample_1["latents"][:, j]] * 2),
                                    torch.cat([sample_1["timesteps"][:, j]] * 2),
                                    embeds_1,
                                ).sample
                                noise_pred_uncond_1, noise_pred_text_1 = noise_pred_1.chunk(2)
                                noise_pred_1 = noise_pred_uncond_1 + config.sample.guidance_scale * (noise_pred_text_1 - noise_pred_uncond_1)

                                noise_ref_pred_1 = ref(
                                    torch.cat([sample_1["latents"][:, j]] * 2),
                                    torch.cat([sample_1["timesteps"][:, j]] * 2),
                                    embeds_1,
                                ).sample
                                noise_ref_pred_uncond_1, noise_ref_pred_text_1 = noise_ref_pred_1.chunk(2)
                                noise_ref_pred_1 = noise_ref_pred_uncond_1 + config.sample.guidance_scale * (
                                    noise_ref_pred_text_1 - noise_ref_pred_uncond_1
                                )

                            else:
                                noise_pred_0 = pipeline.unet(
                                    sample_0["latents"][:, j], sample_0["timesteps"][:, j], embeds_0
                                ).sample
                                noise_ref_pred_0 = ref(
                                    sample_0["latents"][:, j], sample_0["timesteps"][:, j], embeds_0
                                ).sample

                                noise_pred_1 = pipeline.unet(
                                    sample_1["latents"][:, j], sample_1["timesteps"][:, j], embeds_1
                                ).sample
                                noise_ref_pred_1 = ref(
                                    sample_1["latents"][:, j], sample_1["timesteps"][:, j], embeds_1
                                ).sample

                            # compute the log prob of next_latents given latents under the current model
                            _, total_prob_0 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred_0,
                                sample_0["timesteps"][:, j],
                                sample_0["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_0["next_latents"][:, j],
                            )
                            _, total_ref_prob_0 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_ref_pred_0,
                                sample_0["timesteps"][:, j],
                                sample_0["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_0["next_latents"][:, j],
                            )
                            _, total_prob_1 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred_1,
                                sample_1["timesteps"][:, j],
                                sample_1["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_1["next_latents"][:, j],
                            )
                            _, total_ref_prob_1 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_ref_pred_1,
                                sample_1["timesteps"][:, j],
                                sample_1["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_1["next_latents"][:, j],
                            )
                    # clip the probs of the pre-trained model and this model
                    ratio_0 = torch.clamp(torch.exp(total_prob_0-total_ref_prob_0),1 - config.train.eps, 1 + config.train.eps)
                    ratio_1 = torch.clamp(torch.exp(total_prob_1-total_ref_prob_1),1 - config.train.eps, 1 + config.train.eps)
                    loss = -torch.log(torch.sigmoid(config.train.beta*(torch.log(ratio_0))*human_prefer[:,0] + config.train.beta*(torch.log(ratio_1))*human_prefer[:, 1])).mean()

                    # backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_layers.parameters(), config.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()


if __name__ == "__main__":
    app.run(main)

# accelerate launch scripts/finetune_train.py \
# --domain="Inpainting" \
# --type_polyp="sessile" \
# --base_model="bdbao/stable-diffusion-inpainting-polyps-nonLoRA-sessile" \
# --patched_id="logs/using/checkpoints" \
# --sample_pkl_path="data/using" \
# --feedbacks_json_path="data/using/json" \
# --save_dir="logs"
