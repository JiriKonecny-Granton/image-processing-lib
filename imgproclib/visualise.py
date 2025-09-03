"""
visualise.py

This module provides utilities for visualizing images, especially in Jupyter Notebooks.
It includes functions for combining multiple image pages vertically, displaying images
with optional titles and resizing, and creating thumbnails.

Dependencies:
    - numpy
    - Pillow (PIL)
    - IPython.display
    - matplotlib (for potential future use)
    - GrtUtils.convertors
    - MimeDetector
    - .convertors
    - .preprocessing

Raises:
    ValueError: For unsupported input types or failed conversions.
    IOError: For file reading errors.
    Exception: For unexpected errors during visualization.

Author: Granton s.r.o.
Date: 2025-09-03
"""

import numpy as np
from PIL import Image
import io
from typing import List, Optional, Union
import IPython.display as Display

from imgproclib.convertors import convert_to_image
from imgproclib.utils import combine_pages_vertically

def show_as_image(
    file: Union[str, bytes, List[bytes], np.ndarray, Image.Image],
    title: Optional[str] = None,
    w: Optional[int] = None,
    h: Optional[int] = None
):
    """
    Display an image or list of images in a Jupyter Notebook with optional title and dimensions.

    Args:
        file (str | bytes | List[bytes] | np.ndarray | PIL.Image.Image): Image source.
            - If list, images are combined vertically.
        title (str, optional): Title to display above the image.
        w (int, optional): Width of the displayed image.
        h (int, optional): Height of the displayed image.

    Returns:
        None

    Raises:
        Exception: If image conversion or display fails.
    """
    try:
        if isinstance(file, list):
            image = combine_pages_vertically(file)
        else:
            image = convert_to_image(file, metl_to_single_image=True)
        img = Display.Image(data=image, alt=title, width=w, height=h)
        if title:
            try:
                from IPython.display import Markdown as MD
                Display.display(MD(f"**{title}**"))
            except Exception:
                # Fallback if Markdown is not available
                Display.display(title)
        Display.display(img)
    except Exception as e:
        Display.display(f"Error when trying to show image: {e}")

def _make_thumb(img_bytes: bytes, max_w: int = 420, max_h: int = 420) -> Image.Image:
    """
    Create a thumbnail PIL Image from image bytes.

    Args:
        img_bytes (bytes): Image data in bytes.
        max_w (int): Maximum width of the thumbnail.
        max_h (int): Maximum height of the thumbnail.

    Returns:
        PIL.Image.Image: Thumbnail image.
    """
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    im.thumbnail((max_w, max_h))
    return im
