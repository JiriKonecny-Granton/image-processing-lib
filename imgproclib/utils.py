from typing import List
import io
import base64
import numpy as np
from PIL import Image

from imgproclib.convertors import image_to_bytes

def _b64img(img_bytes: bytes) -> str:
    """
    Convert image bytes to a base64-encoded string for HTML embedding.

    Args:
        img_bytes (bytes): Image data in bytes.

    Returns:
        str: Base64-encoded string.

    Raises:
        TypeError: If img_bytes is not bytes.
    """
    if not isinstance(img_bytes, bytes):
        raise TypeError("img_bytes must be of type bytes.")
    return base64.b64encode(img_bytes).decode("ascii")

def combine_pages_vertically(pages_bytes: List[bytes]) -> bytes:
    """
    Combine a list of image bytes (PNG/JPG) vertically into a single PNG image (as bytes).

    Args:
        pages_bytes (List[bytes]): List of image bytes to combine.

    Returns:
        bytes: Combined image as PNG bytes.

    Raises:
        ValueError: If the input list is empty or contains invalid image bytes.
        Exception: If image combination fails.
    """
    if not pages_bytes or not isinstance(pages_bytes, list):
        raise ValueError("pages_bytes must be a non-empty list of image bytes.")
    try:
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in pages_bytes]
        width = max(im.width for im in imgs)
        total_h = sum(im.height for im in imgs)
        canvas = Image.new("RGB", (width, total_h), (255, 255, 255))
        y = 0
        for im in imgs:
            # center narrower images
            x = (width - im.width) // 2
            canvas.paste(im, (x, y))
            y += im.height
        out = io.BytesIO()
        canvas.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        raise Exception(f"Failed to combine pages vertically: {e}") from e

def open_image_as(file: str | bytes | Image.Image, output_format='cv') -> np.ndarray | Image.Image | bytes | str:
    """
    Opens an image file or bytes and returns it in the specified format.

    Args:
        file (str | bytes | PIL.Image.Image): Path to the image file, bytes of the image, or a PIL Image.
        output_format (str): Format to return the image in:
            - 'cv': OpenCV (NumPy array, BGR)
            - 'pil': PIL Image
            - 'bin': Image as bytes
            - 'b64': Image as base64 string

    Returns:
        np.ndarray | PIL.Image.Image | bytes | str: Converted image.

    Raises:
        ValueError: If input or output format is not supported.
        IOError: If file cannot be opened.
    """
    import io
    try:
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV (cv2) is required for 'cv' output format.") from e
    from PIL import Image
    import numpy as np

    try:
        if isinstance(file, str):
            image = Image.open(file).convert('RGB')
        elif isinstance(file, bytes):
            image = Image.open(io.BytesIO(file)).convert('RGB')
        elif isinstance(file, Image.Image):
            image = file.convert('RGB')
        else:
            raise ValueError("Input must be a file path (str), bytes, or PIL Image.")
    except Exception as e:
        raise IOError(f"Failed to open image: {e}") from e

    if output_format == 'cv':
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return image_cv
        except Exception as e:
            raise Exception(f"Failed to convert image to OpenCV format: {e}") from e
    elif output_format == 'pil':
        return image
    elif output_format == 'bin':
        return image_to_bytes(image)
    elif output_format == 'b64':
        image_bytes = image_to_bytes(image)
        return base64.b64encode(image_bytes).decode("ascii")
    else:
        raise ValueError("Output format must be 'cv', 'pil', 'bin', or 'b64'.")
