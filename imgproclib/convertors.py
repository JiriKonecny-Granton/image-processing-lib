"""
convertors.py

This module provides conversion utilities for working with images, including:
- Converting images between PIL, OpenCV (NumPy), bytes, and base64 formats.
- Converting images to PDF.
- Opening images from various sources and returning them in different formats.
- Converting files to binary.
- Converting files (including PDFs and images) to image bytes using external convertors.

Dependencies:
    - numpy
    - Pillow (PIL)
    - OpenCV (cv2)
    - GrtUtils.convertors
    - MimeDetector

Raises:
    ValueError: For unsupported input types or failed conversions.
    IOError: For file reading errors.
    Exception: For unexpected errors during conversion.

Author: Granton s.r.o.
Date: 2025-09-03
"""

import numpy as np
from PIL import Image
import os

def image_to_bytes(image: np.ndarray | Image.Image, img_format='png') -> bytes:
    """
    Converts a PIL Image or NumPy (OpenCV) image to bytes in the specified format.

    Args:
        image (PIL.Image.Image or np.ndarray): The image to convert.
        img_format (str): Format to save the image (default 'png').

    Returns:
        bytes: Image in bytes format.

    Raises:
        ValueError: If the image format is not supported or encoding fails.
    """
    import io
    try:
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV (cv2) is required for NumPy array conversion.") from e
    from PIL import Image
    import numpy as np

    if isinstance(image, Image.Image):
        output = io.BytesIO()
        try:
            image.save(output, format=img_format.upper())
        except Exception as e:
            raise ValueError(f"Failed to save PIL image as {img_format}: {e}") from e
        return output.getvalue()
    elif isinstance(image, np.ndarray):
        ext = img_format
        if not ext.startswith('.'):
            ext = '.' + ext
        success, buffer = cv2.imencode(ext, image)
        if not success:
            raise ValueError(f"Image encoding failed for format: {img_format}")
        return buffer.tobytes()
    else:
        raise ValueError(f"Image format not supported: {type(image)}")

def image_to_pdf(image: bytes | np.ndarray | Image.Image, dpi: int = 300) -> bytes:
    """
    Convert an image to PDF format.

    Args:
        image (bytes | np.ndarray | PIL.Image.Image): The image to convert.
        dpi (int): Resolution for the PDF (default 300).

    Returns:
        bytes: PDF file in bytes.

    Raises:
        ValueError: If the image format is not supported.
        Exception: If PDF conversion fails.
    """
    import io
    from PIL import Image
    import numpy as np

    try:
        if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
            image_bytes = image_to_bytes(image)
        elif isinstance(image, bytes):
            image_bytes = image
        else:
            raise ValueError(f"Image format not supported: {type(image)}")
        with Image.open(io.BytesIO(image_bytes)) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            out = io.BytesIO()
            im.save(out, format="PDF", resolution=dpi)
            return out.getvalue()
    except Exception as e:
        raise Exception(f"Failed to convert image to PDF: {e}") from e

def convert_to_binary(file: str | bytes | Image.Image) -> bytes:
    """
    Converts a file to binary format.

    Args:
        file (str | bytes | PIL.Image.Image): File path, bytes, or PIL Image.

    Returns:
        bytes: File content as bytes.

    Raises:
        ValueError: If input type is not supported.
        IOError: If file cannot be read.
    """
    from io import BytesIO
    from PIL import Image

    if isinstance(file, bytes):
        return file
    elif isinstance(file, str):
        try:
            with open(file, 'rb') as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read file '{file}': {e}") from e
    elif isinstance(file, Image.Image):
        buffer = BytesIO()
        try:
            file.save(buffer, format='png')
        except Exception as e:
            raise ValueError(f"Failed to save PIL image as PNG: {e}") from e
        return buffer.getvalue()
    else:
        raise ValueError(f"Unsupported input type for converting to binary: {type(file)}")

from GrtUtils.convertors import Convertors
from MimeDetector import MimeDetector

def convert_to_image(file: str | bytes | np.ndarray | Image.Image, metl_to_single_image=True) -> bytes | list[bytes]:
    """
    Converts a file (path, bytes, numpy array, or PIL Image) to image bytes using Convertors.

    Args:
        file (str | bytes | np.ndarray | PIL.Image.Image): The input file to convert.
        metl_to_single_image (bool): If True, returns single image bytes.
                                     If False, returns list of image bytes (for multi-page documents).

    Returns:
        bytes or list[bytes]: Image bytes or list of image bytes.

    Raises:
        ValueError: If the input type is not supported.
        Exception: If conversion fails.
    """
    try:
        if isinstance(file, str) or isinstance(file, bytes):
            mime_type = MimeDetector.detect_mime_type(file)
        elif isinstance(file, Image.Image):
            mime_type = f"image/{(file.format or 'png').lower()}"
        else:
            raise ValueError(f"Unsupported input type: {type(file)}")

        PDF_CONVERTOR_ENDPOINT:str = os.getenv("PDF_CONVERTOR_ENDPOINT") if os.getenv("PDF_CONVERTOR_ENABLED") is not None else "NO_ENDPOINT"
        binary_file = convert_to_binary(file)
        if metl_to_single_image:
            return Convertors.convert_to_image(
                binary_file,
                document_image_render_format='png',
                pdf_convertor_endpoint=PDF_CONVERTOR_ENDPOINT,
                mime_type=mime_type
            )
        else:
            return Convertors.convert_to_images(
                binary_file,
                document_image_render_format='png',
                pdf_convertor_endpoint=PDF_CONVERTOR_ENDPOINT,
                mime_type=mime_type
            )
    except Exception as e:
        raise Exception(f"Failed to convert file to image: {e}") from e