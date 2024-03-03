from importlib import resources
import os
import functools
import random
import time
import inflect
from PIL import Image

IE = inflect.engine()
ASSETS_PATH = resources.files("d3po_pytorch.assets")
IMAGES_PATH = resources.files("train_data")

@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `d3po_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

@functools.cache
def _load_images(path, mask_path):
    """
    Load PNG images from the specified directory path and return them as a list of PIL.Image.Image objects.
    First tries to load from `path` directly, and if that doesn't exist, searches the
    `d3po_pytorch/assets` directory for a directory named `path`.
    """
    image_list = []

    if not os.path.exists(path):
        newpath = IMAGES_PATH.joinpath(path)
        newmaskpath = IMAGES_PATH.joinpath(mask_path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or d3po_pytorch.assets/{path}")

    path = newpath
    mask_path = newmaskpath

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
    prompts = []
    if image == False:
        prompts = _load_lines(path)[low:high]
    else: 
        prompts = _load_images(path, mask)[low:high]
    return random.choice(prompts), {}

def kvasir_imgs():
    # return from_file("kvasir/sessile-polyps/images", image = True, mask = "kvasir/sessile-polyps/masks")

    return _load_images("kvasir/sessile-polyps/images", "kvasir/sessile-polyps/masks") # 20 first images

def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def simple_animal():
    return from_file("simple_animals.txt")


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def kvasir_prompt():
    return from_file("kvasir.txt")

def anything_prompt():
    return from_file("anything_prompt.txt")

def unsafe_prompt():
    return from_file("unsafe_prompt.txt")

if __name__ == "__main__":
    print(len(_load_images("kvasir/sessile-polyps/masks")))
