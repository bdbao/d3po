from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")
prompt = "sessile polyps"
image = Image.open("../train_data/kvasir/sessile-polyps/images/cju0qoxqj9q6s0835b43399p4.jpg")
masked = Image.open("../train_data/kvasir/sessile-polyps/masks/cju0qoxqj9q6s0835b43399p4.jpg")
image.show()
masked.show()
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=masked).images[0]
image.save("./test.png")
