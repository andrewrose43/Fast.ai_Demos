## IMPORTS

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from PIL import Image


## These functions adapted from my notebook Ch10_NegativePrompts.ipynb

def stable_diffusion_basic_components():
    '''
    Returns a tuple containing some basic components used repeatedly throughout the fast.ai course.
    '''
    
    beta_start, beta_end = 0.00085, 0.012
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to("cuda")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to("cuda")
    lms_scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000)

    return (tokenizer, text_encoder, vae, unet, lms_scheduler)


def tokenize_text(tokenizer, text):
    '''
    Uses the input tokenizer to cut a string into tokens.
    Returns the tokens in the familiar format used in the fast.ai course.
    '''
    return tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")


def encode_tokens(text_encoder, tokens):
    '''
    Takes tokens from tokenize_text() and encodes them as vectors for use in CLIP.
    '''
    return text_encoder(tokens.input_ids.to("cuda"))[0].half()


def generate_noise_latents(unet, scheduler, seed, height, width, batch_size=1):
    '''
    Returns a set of noise latents of the right specifications, both for the image size and the scheduler.
    '''
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn((batch_size, unet.in_channels, height // 8, width // 8)).to("cuda").half() * scheduler.init_noise_sigma


def latents_to_image(latents, vae):
    '''
    Convert latents to a full-resolution image.
    '''
    with torch.no_grad(): image = vae.decode(1 / 0.18215 * latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)  # Make everything between 0 and 1
    image = image[0].detach().cpu().permute(1, 2, 0).numpy()  # Put it back on CPU and ensure order of dimensions is what Python imaging expects
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)