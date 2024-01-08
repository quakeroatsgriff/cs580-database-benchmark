import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
import time 

class ImageGenerator():
  def __init__(self, img_height = 10, img_width = 10, num_inferences_steps = 100, batch_size = 10, seed_num = round( time.time() ) ):
    """
    Initialize model stuff so it is loaded in memory
    Parameters:
      - Image Height
      - Image Width
      - Number of inference steps (epochs)
      - Image generation batch size
      - Seed number for latent RNG
    """
    self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the autoencoder model which will be used to decode the latents into image space. 
    self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # Load the self.tokenizer and text encoder to tokenize and encode the text. 
    self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    #The self.unet model for generating the latents.
    self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    self.scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    self.vae = self.vae.to(self.torch_device)
    self.text_encoder = self.text_encoder.to(self.torch_device)
    self.unet = self.unet.to(self.torch_device) 

    self.batch_size = batch_size
    
    # default for Stable Diffusion dimensions are 512 x 512
    self.img_height = img_height
    self.img_width = img_width
    
    # Number of denoising steps
    self.num_inferences_steps = num_inferences_steps
    
    # Scale for classifier-free guidance
    self.guidance_scale = 7.5

    # self.prompt = ["DEFAULT"]
    
    # Seed generator to create the inital latent noise
    rng = torch.manual_seed(seed_num)   

    self.latents = torch.randn(  
      (batch_size, self.unet.in_channels, self.img_height // 8, self.img_width // 8),
      generator=rng,
    )
    self.latents = self.latents.to(self.torch_device)

    self.scheduler.set_timesteps(self.num_inferences_steps)

    self.latents = self.latents * self.scheduler.init_noise_sigma

  def new_seed(self):
    seed_num = round( time.time() * 100 )
    rng = torch.manual_seed(seed_num) 

    self.latents = torch.randn(  
      (self.batch_size, self.unet.in_channels, self.img_height // 8, self.img_width // 8),
      generator=rng,
    )
    self.latents = self.latents.to(self.torch_device)

    self.scheduler.set_timesteps(self.num_inferences_steps)

    self.latents = self.latents * self.scheduler.init_noise_sigma
    pass

  def get_text_embedding(self, prompts):
    """Gets embeddigs:
    Parameters: 
    TODO
    """
    text_input = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
      text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]
      max_length = text_input.input_ids.shape[-1]

    uncond_input = self.tokenizer([""] * self.batch_size, padding="max_length", max_length=max_length, return_tensors="pt")

    with torch.no_grad():
      uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]   

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings
  
  def compute_noise(self, text_embeddings):
    """Takes list of prompts and creates images. num of prompts must equal batch size.
    Parameters: 
      - Text embeddings from encoder
    """

    for t in tqdm(self.scheduler.timesteps):
      # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
      latent_model_input = torch.cat([self.latents] * 2)

      latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

      # predict the noise residual
      with torch.no_grad():
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states = text_embeddings).sample

      # perform guidance
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

      # compute the previous noisy sample x_t -> x_t-1
      self.latents = self.scheduler.step(noise_pred, t, self.latents).prev_sample
    
    pass
  
  def package_generated_images(self):
    """Moves the image from CUDA to CPU cache. Also decodes the latent image back into pixel space."""
    # scale and decode the image latents with self.vae
    self.latents = 1 / 0.18215 * self.latents
    with torch.no_grad():
      image = self.vae.decode(self.latents).sample

      image = (image / 2 + 0.5).clamp(0, 1)
      image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
      images = (image * 255).round().astype("uint8")
      pil_images = [Image.fromarray(image) for image in images]
      return pil_images

  def generate_images( self, prompts = ["Test Image"] ):
    """
    This is the function you interact with to generate images.
    Parameters:
      - Prompts: A list of strings. THE LENGTH OF PROMPTS MUST BE EQUAL TO
      THE BATCH SIZE!
    """
    assert len(prompts) == self.batch_size

    text_embeddings = self.get_text_embedding( prompts )
    self.compute_noise( text_embeddings )
    images = self.package_generated_images()
    return images

def main():
  print("TESTING IMAGE GENERATOR WORKS: PLEASE USE main.py FOR MORE FUNCTIONALITY")
  ig = ImageGenerator(batch_size=1, num_inferences_steps = 10)
  pil_images = ig.generate_images()

  for i,img in enumerate(pil_images):
    img.save(f'{i}.png')
    print(f"Created {i}.png")
  pass

if __name__ == "__main__":
  main()


