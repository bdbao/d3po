import contextlib
import os
import datetime
# import torchvision
import time
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import d3po_pytorch.prompts
import d3po_pytorch.rewards
from d3po_pytorch.diffusers_patch.pipeline_with_logprob_inpaint import pipeline_with_logprob_inpaint
import torch
from functools import partial
import tqdm
from PIL import Image
import json
import pickle
from scripts.utils import post_processing

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_dir = './data'
save_dir = os.path.join(save_dir, now_time)

FLAGS = flags.FLAGS
NUM_PER_PROMPT = 7
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        print("loading model. Please Wait.")
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        print("load successfully!")

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
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )

    # set a random seed
    ramdom_seed = np.random.randint(0,100000)
    set_seed(ramdom_seed, device_specific=True)

    # load scheduler, tokenizer and models.
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

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
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
    # prepare prompt and reward fn
    prompt_fn = getattr(d3po_pytorch.prompts, config.prompt_fn)
    image_fn = getattr(d3po_pytorch.prompts, config.image_fn)
    
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

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)

    #################### SAMPLING ####################
    pipeline.unet.eval()
    samples = []
    total_prompts = []
    for i in tqdm(
        range(config.sample.num_batches_per_epoch),
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        # generate prompts
        prompts1, prompt_metadata = zip(
            *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
        )

        masks = []
        input_images = []
        for _ in range(config.sample.batch_size):
            im, m = image_fn()[0]
            masks.append(m)
            input_images.append(im)

            # pil_image_im = torchvision.transforms.ToPILImage()(im)
            # pil_image_m = torchvision.transforms.ToPILImage()(m)

            # pil_image_im.save('/content/im.jpg')
            # pil_image_m.save('/content/m.jpg')
            # exit()

            # print(m.cpu().numpy().dtype())

            print('type(im):', type(im), ", shape:", im.shape)
            # pil = Image.fromarray((m.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"), "RGB")
            # pil.save('/content/pil.jpg')
            # pil.show()
            # exit(0)

        masks = torch.stack(masks)
        print('type(input_images)', type(input_images), ", shape:", np.asarray(input_images).shape)
        input_images = torch.stack(input_images)
        print('type(input_images) after torch.stack(input_images)', type(input_images), ", shape:", input_images.shape)

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
            print("Before detach:")
            # pil = Image.fromarray((images1.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"), "RGB")
            # pil.save('/content/images1.jpg')
            print('type(images1)', type(images1), ", shape:", images1.shape)

            latents1 = torch.stack(latents1, dim=1)
            images1 = images1.cpu().detach()
            latents1 = latents1.cpu().detach()

            print("After detach:")
            # pil1 = Image.fromarray((images1.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"), "RGB")
            # pil1.save('/content/images1_1.jpg')
            print('type(images1)', type(images1), ", shape:", images1.shape)

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
        print('samples.shape', np.asarray(samples[0]['images']).shape)

        # for j, image in enumerate(images):
        #   for k in range(NUM_PER_PROMPT):
        #       print('image.shape', np.asarray(image).shape)
        #       print('image[k].shape', np.asarray(image[k]).shape)
        #       pil = Image.fromarray((image[k].cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"), "RGB")
        #       pil.save(os.path.join(save_dir, f"images/{(NUM_PER_PROMPT*j+global_idx+k):05}.jpg"))

        # exit()

        os.makedirs(os.path.join(save_dir, "images/"), exist_ok=True)
        if (i+1)%config.sample.save_interval ==0 or i==(config.sample.num_batches_per_epoch-1):
            print(f'-----------{accelerator.process_index} save image start-----------')
            print("\n\n\n Sample:\n\n\n", samples[0].keys())
            print('len(samples)', len(samples))
            exit(0)
            new_samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            # new_samples = samples
            print("\n\n\n new_samples:\n\n\n", new_samples.keys())
            print('new_samples.shape', np.asarray(new_samples['images']).shape)
            images = new_samples['images'][local_idx:]
            print("new_samples['images'][local_idx:].shape", np.asarray(images).shape)
            for j, image in enumerate(images):
                for k in range(NUM_PER_PROMPT):
                    print('image.shape', np.asarray(image).shape)
                    print('image[k].shape', np.asarray(image[k]).shape)
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
        print(f'GPU: {accelerator.device} done')
    if accelerator.is_main_process:
        while True:
            done = [True if os.path.exists(os.path.join(save_dir, f'{i}.txt')) else False for i in range(accelerator.num_processes)]
            if all(done):
                time.sleep(5)
                break
        print('---------start post processing---------')
        post_processing(save_dir, accelerator.num_processes)

if __name__ == "__main__":
    app.run(main)



'''
2024-02-20 15:01:38.679149: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-02-20 15:01:38.679203: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-02-20 15:01:38.686040: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-02-20 15:01:40.682114: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
The following values were not passed to `accelerate launch` and had defaults used instead:
	`--num_processes` was set to a value of `1`
	`--num_machines` was set to a value of `1`
	`--mixed_precision` was set to a value of `'no'`
	`--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
2024-02-20 15:01:46.449269: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-02-20 15:01:46.449319: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-02-20 15:01:46.450497: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-02-20 15:01:47.725127: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
The config attributes {'image_encoder': [None, None]} were passed to StableDiffusionInpaintPipeline, but are not expected and will be ignored. Please verify your model_index.json configuration file.
Keyword arguments {'image_encoder': [None, None]} are not expected by StableDiffusionInpaintPipeline and will be ignored.
Some weights of StableDiffusionSafetyChecker were not initialized from the model checkpoint at /root/.cache/huggingface/hub/models--bdbao--stable-diffusion-inpainting-polyps-nonLoRA-sessile/snapshots/60b1d3e3f189d4adbc910ea0a093ca28a9286c60/safety_checker and are newly initialized: ['vision_model.vision_model.embeddings.position_ids']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
The config attributes {'addition_time_embed_dim': None, 'attention_type': 'default', 'dropout': 0.0, 'num_attention_heads': None, 'reverse_transformer_layers_per_block': None, 'transformer_layers_per_block': 1} were passed to UNet2DConditionModel, but are not expected and will be ignored. Please verify your config.json configuration file.
The config attributes {'force_upcast': True} were passed to AutoencoderKL, but are not expected and will be ignored. Please verify your config.json configuration file.
  0% 0/1 [00:00<?, ?it/s]type(im): <class 'torch.Tensor'> , shape: torch.Size([3, 512, 512])
type(input_images) <class 'list'> , shape: (1, 3, 512, 512)
type(input_images) after torch.stack(input_images) <class 'torch.Tensor'> , shape: torch.Size([1, 3, 512, 512])

Timestep:   0% 0/20 [00:00<?, ?it/s]
Timestep:   5% 1/20 [00:00<00:05,  3.18it/s]
Timestep:  10% 2/20 [00:00<00:04,  4.01it/s]
Timestep:  15% 3/20 [00:00<00:03,  4.40it/s]
Timestep:  20% 4/20 [00:00<00:03,  4.60it/s]
Timestep:  25% 5/20 [00:01<00:03,  4.62it/s]
Timestep:  30% 6/20 [00:01<00:03,  4.65it/s]
Timestep:  35% 7/20 [00:01<00:02,  4.74it/s]
Timestep:  40% 8/20 [00:01<00:02,  4.82it/s]
Timestep:  45% 9/20 [00:01<00:02,  4.81it/s]
Timestep:  50% 10/20 [00:02<00:02,  4.74it/s]
Timestep:  55% 11/20 [00:02<00:01,  4.79it/s]
Timestep:  60% 12/20 [00:02<00:01,  4.81it/s]
Timestep:  65% 13/20 [00:02<00:01,  4.87it/s]
Timestep:  70% 14/20 [00:02<00:01,  4.88it/s]
Timestep:  75% 15/20 [00:03<00:01,  4.85it/s]
Timestep:  80% 16/20 [00:03<00:00,  4.80it/s]
Timestep:  85% 17/20 [00:03<00:00,  4.82it/s]
Timestep:  90% 18/20 [00:03<00:00,  4.86it/s]
Timestep:  95% 19/20 [00:04<00:00,  4.86it/s]
Timestep: 100% 20/20 [00:04<00:00,  4.82it/s]
                                             Before detach:
type(images1) <class 'torch.Tensor'> , shape: torch.Size([1, 3, 512, 512])
After detach:
type(images1) <class 'torch.Tensor'> , shape: torch.Size([1, 3, 512, 512])

Timestep:   0% 0/20 [00:00<?, ?it/s]
Timestep:   5% 1/20 [00:00<00:05,  3.41it/s]
Timestep:  10% 2/20 [00:00<00:04,  4.06it/s]
Timestep:  15% 3/20 [00:00<00:03,  4.40it/s]
Timestep:  20% 4/20 [00:00<00:03,  4.54it/s]
Timestep:  25% 5/20 [00:01<00:03,  4.61it/s]
Timestep:  30% 6/20 [00:01<00:02,  4.71it/s]
Timestep:  35% 7/20 [00:01<00:02,  4.77it/s]
Timestep:  40% 8/20 [00:01<00:02,  4.77it/s]
Timestep:  45% 9/20 [00:01<00:02,  4.74it/s]
Timestep:  50% 10/20 [00:02<00:02,  4.80it/s]
Timestep:  55% 11/20 [00:02<00:01,  4.82it/s]
Timestep:  60% 12/20 [00:02<00:01,  4.80it/s]
Timestep:  65% 13/20 [00:02<00:01,  4.76it/s]
Timestep:  70% 14/20 [00:03<00:01,  4.74it/s]
Timestep:  75% 15/20 [00:03<00:01,  4.75it/s]
Timestep:  80% 16/20 [00:03<00:00,  4.78it/s]
Timestep:  85% 17/20 [00:03<00:00,  4.77it/s]
Timestep:  90% 18/20 [00:03<00:00,  4.76it/s]
Timestep:  95% 19/20 [00:04<00:00,  4.77it/s]
Timestep: 100% 20/20 [00:04<00:00,  4.74it/s]
                                             
Timestep:   0% 0/20 [00:00<?, ?it/s]
Timestep:   5% 1/20 [00:00<00:05,  3.32it/s]
Timestep:  10% 2/20 [00:00<00:04,  4.02it/s]
Timestep:  15% 3/20 [00:00<00:03,  4.33it/s]
Timestep:  20% 4/20 [00:00<00:03,  4.44it/s]
Timestep:  25% 5/20 [00:01<00:03,  4.52it/s]
Timestep:  30% 6/20 [00:01<00:03,  4.61it/s]
Timestep:  35% 7/20 [00:01<00:02,  4.63it/s]
Timestep:  40% 8/20 [00:01<00:02,  4.63it/s]
Timestep:  45% 9/20 [00:02<00:02,  4.67it/s]
Timestep:  50% 10/20 [00:02<00:02,  4.65it/s]
Timestep:  55% 11/20 [00:02<00:01,  4.68it/s]
Timestep:  60% 12/20 [00:02<00:01,  4.67it/s]
Timestep:  65% 13/20 [00:02<00:01,  4.66it/s]
Timestep:  70% 14/20 [00:03<00:01,  4.68it/s]
Timestep:  75% 15/20 [00:03<00:01,  4.68it/s]
Timestep:  80% 16/20 [00:03<00:00,  4.68it/s]
Timestep:  85% 17/20 [00:03<00:00,  4.66it/s]
Timestep:  90% 18/20 [00:03<00:00,  4.67it/s]
Timestep:  95% 19/20 [00:04<00:00,  4.67it/s]
Timestep: 100% 20/20 [00:04<00:00,  4.69it/s]
                                             
Timestep:   0% 0/20 [00:00<?, ?it/s]
Timestep:   5% 1/20 [00:00<00:05,  3.26it/s]
Timestep:  10% 2/20 [00:00<00:04,  3.99it/s]
Timestep:  15% 3/20 [00:00<00:03,  4.31it/s]
Timestep:  20% 4/20 [00:00<00:03,  4.44it/s]
Timestep:  25% 5/20 [00:01<00:03,  4.53it/s]
Timestep:  30% 6/20 [00:01<00:03,  4.58it/s]
Timestep:  35% 7/20 [00:01<00:02,  4.58it/s]
Timestep:  40% 8/20 [00:01<00:02,  4.62it/s]
Timestep:  45% 9/20 [00:02<00:02,  4.59it/s]
Timestep:  50% 10/20 [00:02<00:02,  4.57it/s]
Timestep:  55% 11/20 [00:02<00:01,  4.57it/s]
Timestep:  60% 12/20 [00:02<00:01,  4.57it/s]
Timestep:  65% 13/20 [00:02<00:01,  4.61it/s]
Timestep:  70% 14/20 [00:03<00:01,  4.62it/s]
Timestep:  75% 15/20 [00:03<00:01,  4.60it/s]
Timestep:  80% 16/20 [00:03<00:00,  4.63it/s]
Timestep:  85% 17/20 [00:03<00:00,  4.62it/s]
Timestep:  90% 18/20 [00:03<00:00,  4.58it/s]
Timestep:  95% 19/20 [00:04<00:00,  4.57it/s]
Timestep: 100% 20/20 [00:04<00:00,  4.57it/s]
                                             
Timestep:   0% 0/20 [00:00<?, ?it/s]
Timestep:   5% 1/20 [00:00<00:05,  3.29it/s]
Timestep:  10% 2/20 [00:00<00:04,  3.93it/s]
Timestep:  15% 3/20 [00:00<00:04,  4.18it/s]
Timestep:  20% 4/20 [00:00<00:03,  4.30it/s]
Timestep:  25% 5/20 [00:01<00:03,  4.43it/s]
Timestep:  30% 6/20 [00:01<00:03,  4.51it/s]
Timestep:  35% 7/20 [00:01<00:02,  4.51it/s]
Timestep:  40% 8/20 [00:01<00:02,  4.52it/s]
Timestep:  45% 9/20 [00:02<00:02,  4.56it/s]
Timestep:  50% 10/20 [00:02<00:02,  4.56it/s]
Timestep:  55% 11/20 [00:02<00:01,  4.57it/s]
Timestep:  60% 12/20 [00:02<00:01,  4.52it/s]
Timestep:  65% 13/20 [00:02<00:01,  4.50it/s]
Timestep:  70% 14/20 [00:03<00:01,  4.50it/s]
Timestep:  75% 15/20 [00:03<00:01,  4.50it/s]
Timestep:  80% 16/20 [00:03<00:00,  4.52it/s]
Timestep:  85% 17/20 [00:03<00:00,  4.53it/s]
Timestep:  90% 18/20 [00:04<00:00,  4.52it/s]
Timestep:  95% 19/20 [00:04<00:00,  4.50it/s]
Timestep: 100% 20/20 [00:04<00:00,  4.51it/s]
                                             
Timestep:   0% 0/20 [00:00<?, ?it/s]
Timestep:   5% 1/20 [00:00<00:05,  3.22it/s]
Timestep:  10% 2/20 [00:00<00:04,  3.85it/s]
Timestep:  15% 3/20 [00:00<00:04,  4.07it/s]
Timestep:  20% 4/20 [00:00<00:03,  4.20it/s]
Timestep:  25% 5/20 [00:01<00:03,  4.32it/s]
Timestep:  30% 6/20 [00:01<00:03,  4.40it/s]
Timestep:  35% 7/20 [00:01<00:02,  4.41it/s]
Timestep:  40% 8/20 [00:01<00:02,  4.40it/s]
Timestep:  45% 9/20 [00:02<00:02,  4.44it/s]
Timestep:  50% 10/20 [00:02<00:02,  4.47it/s]
Timestep:  55% 11/20 [00:02<00:02,  4.49it/s]
Timestep:  60% 12/20 [00:02<00:01,  4.47it/s]
Timestep:  65% 13/20 [00:02<00:01,  4.45it/s]
Timestep:  70% 14/20 [00:03<00:01,  4.46it/s]
Timestep:  75% 15/20 [00:03<00:01,  4.47it/s]
Timestep:  80% 16/20 [00:03<00:00,  4.49it/s]
Timestep:  85% 17/20 [00:03<00:00,  4.51it/s]
Timestep:  90% 18/20 [00:04<00:00,  4.50it/s]
Timestep:  95% 19/20 [00:04<00:00,  4.52it/s]
Timestep: 100% 20/20 [00:04<00:00,  4.53it/s]
                                             
Timestep:   0% 0/20 [00:00<?, ?it/s]
Timestep:   5% 1/20 [00:00<00:05,  3.23it/s]
Timestep:  10% 2/20 [00:00<00:04,  3.87it/s]
Timestep:  15% 3/20 [00:00<00:04,  4.17it/s]
Timestep:  20% 4/20 [00:00<00:03,  4.24it/s]
Timestep:  25% 5/20 [00:01<00:03,  4.32it/s]
Timestep:  30% 6/20 [00:01<00:03,  4.38it/s]
Timestep:  35% 7/20 [00:01<00:02,  4.41it/s]
Timestep:  40% 8/20 [00:01<00:02,  4.45it/s]
Timestep:  45% 9/20 [00:02<00:02,  4.45it/s]
Timestep:  50% 10/20 [00:02<00:02,  4.47it/s]
Timestep:  55% 11/20 [00:02<00:02,  4.47it/s]
Timestep:  60% 12/20 [00:02<00:01,  4.47it/s]
Timestep:  65% 13/20 [00:02<00:01,  4.45it/s]
Timestep:  70% 14/20 [00:03<00:01,  4.47it/s]
Timestep:  75% 15/20 [00:03<00:01,  4.49it/s]
Timestep:  80% 16/20 [00:03<00:00,  4.48it/s]
Timestep:  85% 17/20 [00:03<00:00,  4.47it/s]
Timestep:  90% 18/20 [00:04<00:00,  4.47it/s]
Timestep:  95% 19/20 [00:04<00:00,  4.47it/s]
Timestep: 100% 20/20 [00:04<00:00,  4.50it/s]
                                             samples.shape (1, 7, 3, 512, 512)
-----------0 save image start-----------
new_samples.shape (1, 7, 3, 512, 512)
image[k].shape (3, 512, 512)
image[k].shape (3, 512, 512)
image[k].shape (3, 512, 512)
image[k].shape (3, 512, 512)
image[k].shape (3, 512, 512)
image[k].shape (3, 512, 512)
image[k].shape (3, 512, 512)
100% 1/1 [00:37<00:00, 37.82s/it]
GPU: cuda done
---------start post processing---------
data save dir: ./data/2024-02-20-15-01-51
---------write prompt---------
---------write sample---------
---------start check---------
---------start remove---------
mid file:  ['prompt0.json', 'sample0.pkl', '0.txt']
prompt0.json delete successfully!
sample0.pkl delete successfully!
0.txt delete successfully!
'''