# Original implementation by https://github.com/sniklaus/pytorch-hed

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.cuda

from .network import Network


def _PIL2tensor(img: PIL.Image) -> torch.Tensor:
    """Given a Pillow Image, returns a tensor 
    to be fed to the network.

    Args:
        img (PIL.Image)

    Returns:
        torch.Tensor
    """
    array = np.array(img)
    array = array[:, :, ::-1].transpose(2, 0, 1)
    array = array.astype(np.float32) * (1.0 / 255.0)
    array = np.ascontiguousarray(array)
    tensor = torch.FloatTensor(array)
    return tensor


def _tensor2PIL(tensor: torch.Tensor) -> PIL.Image:
    """Given a tensor (e.g. the network's output),
    returns it converted in a Pillow Image

    Args:
        tensor (torch.Tensor)

    Returns:
        PIL.Image
    """
    tensor = tensor.clamp(0.0, 1.0)
    array = tensor.numpy()
    array = array.transpose(1, 2, 0)[:, :, 0] * 255.0
    array = array.astype(np.uint8)
    img = PIL.Image.fromarray(array)
    return img


def _load_imagelist(dir: str) -> list[str]:
    """Given a directory, loads all the 
    ``.jpg``, ``.gif``, ``.png``, and ``.tga`` 
    files and returns a path list. 

    Args:
        dir (str): The images directory

    Returns:
        list[str]: The images path list
    """
    imgs = []
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for f in os.listdir(dir):
        f_path = os.path.join(dir, f)
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        try:
            _ = PIL.Image.open(f_path)
            imgs.append(f_path)
        except IOError as e:
            warnings.warn(f"Warning, {f_path} not a supported image: {e}")
    return imgs


def _estimate(tensor_in: torch.Tensor, use_cuda : bool = False) -> torch.Tensor:
    """Applies HED to an input tensor containing an image,
    returning a image tensor containing the _estimated edges

    Args:
        tensor_in (torch.Tensor)
        use_cuda (bool): Use CUDA for processing images. Defaults to False

    Returns:
        torch.Tensor
    """

    # requires at least pytorch version 1.3.0
    assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13)
    # make sure to not compute gradients for computational performance
    torch.set_grad_enabled(False)
    # make sure to use cudnn for computational performance
    torch.backends.cudnn.enabled = True

    # Load the network
    net = Network()
    if use_cuda and torch.cuda.is_available():
        net.cuda()
    net.eval()

    width_in = tensor_in.shape[2]
    height_in = tensor_in.shape[1]

    if width_in != 480 or height_in == 320:
        warnings.warn(
            f"Warning, image size (W,H)=({width_in},{height_in})," +
            " there is no guarantee for correctness for (W,H)" +
            " different from (480,320)")

    if use_cuda and torch.cuda.is_available():
        tensor_in = tensor_in.cuda()

    tensor_in = tensor_in.view(1, 3, height_in, width_in)
    tensor_out = net(tensor_in)
    tensor_out = tensor_out[0, :, :, :].cpu()

    return tensor_out


def process_img(img: PIL.Image) -> PIL.Image:
    """Given a Pillow image object, applies HED to it
    and returns the processed PIL.Image

    Args:
        img (PIL.Image): Input image
    
    Returns:
        PIL.Image: Output Image
    """
    # Img -> tensor
    tensor_in = _PIL2tensor(img)
    # Estimation
    tensor_out = _estimate(tensor_in)
    # Img <- tensor
    img = _tensor2PIL(tensor_out)
    return img

def process_file(input_fn: str, output_fn: str) -> None:
    """Given an image file, applies HED to it and 
    writes the output in another image

    Args:
        input_fn (str): Input image filename
        output_fn (str): Output image filename
    """
    # Load image
    img = PIL.Image.open(input_fn)
    # Process img
    img = process_img(img)    
    # Store img
    img.save(output_fn)


def process_folder(input_dir: str, output_dir: str) -> None:
    """Given a directory, applies to it HED to all
    images in it and store the output images in another directory

    Args:
        input_dir (str): Input directory
        output_dir (str): Output directory
    """
    imagelist = _load_imagelist(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for path in imagelist:
        name = Path(path).name
        process_file(path, os.path.join(output_dir, name))
