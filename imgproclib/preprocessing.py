"""
preprocessing.py

This module provides advanced image preprocessing utilities for document images,
including highlighter suppression, deskewing, background normalization, binarization,
noise removal, and batch preprocessing pipelines. It is designed for robust OCR and
document analysis workflows.

Features:
    - Highlighter suppression in color images.
    - Deskewing of scanned documents.
    - Background normalization (Gaussian and median).
    - Smart grayscale conversion.
    - Multiple binarization methods (threshold, Otsu, adaptive, Sauvola, blackhat).
    - Noise and speckle removal.
    - Batch and parallel preprocessing of multi-page documents.
    - Line detection and removal.
    - Integration with external convertors for PDF/image conversion.

Dependencies:
    - numpy
    - opencv-python (cv2)
    - concurrent.futures
    - GrtUtils.convertors
    - MimeDetector
    - .convertors

Raises:
    ValueError: For unsupported input types or failed conversions.
    IOError: For file reading errors.
    Exception: For unexpected errors during preprocessing.

Author: Granton s.r.o.
Date: 2025-09-03

Usage example:
    >>> from preprocessing import convert_and_preprocess_document
    >>> processed_pages = convert_and_preprocess_document("document.pdf", method="otsu")
    >>> # processed_pages is a list of bytes, each representing a preprocessed image page
"""

from typing import Optional, List, Tuple, Callable
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from GrtUtils.convertors import Convertors
from MimeDetector import MimeDetector

from imgproclib.convertors import convert_to_binary, image_to_bytes
from imgproclib.utils import open_image_as

# --- Utility functions ---

def suppress_highlighter_hsv(
    bgr: np.ndarray,
    s_min: int = 70,
    v_min: int = 150,
    hue_ranges: Tuple[Tuple[int, int], ...] = ((15,45), (40,85), (145,180), (0,5))
) -> np.ndarray:
    """
    Suppress yellow/green/pink highlighter marks in a BGR image using HSV masking.

    Args:
        bgr (np.ndarray): Input BGR image.
        s_min (int): Minimum saturation for highlighter detection.
        v_min (int): Minimum value for highlighter detection.
        hue_ranges (tuple): Tuple of hue ranges for highlighter detection.

    Returns:
        np.ndarray: BGR image with suppressed highlighter marks.

    Raises:
        ValueError: If input is not a valid BGR image.
    """
    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Input must be a BGR image (3-channel numpy array).")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    sat_mask = S >= s_min
    val_mask = V >= v_min
    hue_mask = np.zeros_like(S, dtype=bool)
    for lo, hi in hue_ranges:
        if lo <= hi:
            hue_mask |= ((H >= lo) & (H <= hi))
        else:
            hue_mask |= ((H >= lo) | (H <= hi))
    m = (sat_mask & val_mask & hue_mask)
    S = S.copy(); V = V.copy()
    S[m] = 0
    V[m] = 255
    hsv2 = cv2.merge([H, S, V])
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def _deskew(gray: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Deskew an image by detecting and correcting the skew angle.

    Args:
        gray (np.ndarray): Input grayscale image.
        max_angle (float): Maximum angle (in degrees) for deskewing.

    Returns:
        np.ndarray: Deskewed image.

    Raises:
        ValueError: If input is not a valid grayscale image.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input must be a grayscale image (2D numpy array).")
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    if lines is None:
        return gray
    angles = []
    for rho, theta in lines[:,0]:
        angle = (theta * 180/np.pi) - 90  # horizontal text ~ 0°
        if -max_angle <= angle <= max_angle:
            angles.append(angle)
    if not angles:
        return gray
    angle = float(np.median(angles))
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _bg_normalize(gray: np.ndarray, sigma: float = 35.0) -> np.ndarray:
    """
    Normalize the background of an image using Gaussian blur and division.

    Args:
        gray (np.ndarray): Input grayscale image.
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        np.ndarray: Background-normalized image.

    Raises:
        ValueError: If input is not a valid grayscale image.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input must be a grayscale image (2D numpy array).")
    bg = cv2.GaussianBlur(gray, (0,0), sigma)
    norm = cv2.divide(gray, bg, scale=128)
    return norm

def _remove_speckles(bw: np.ndarray, min_area: int = 12) -> np.ndarray:
    """
    Remove small speckles from a binary image using connected components.

    Args:
        bw (np.ndarray): Input binary image.
        min_area (int): Minimum area for connected components to keep.

    Returns:
        np.ndarray: Binary image with removed speckles.

    Raises:
        ValueError: If input is not a valid binary image.
    """
    if not isinstance(bw, np.ndarray) or bw.ndim != 2:
        raise ValueError("Input must be a binary image (2D numpy array).")
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(bw)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def _odd(n: int) -> int:
    """Ensure the number is odd (for kernel sizes)."""
    return n if n % 2 == 1 else n + 1

def _bg_normalize_median(gray: np.ndarray) -> np.ndarray:
    """
    Normalize the background of an image using median blur and division.

    Args:
        gray (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Background-normalized image.

    Raises:
        ValueError: If input is not a valid grayscale image.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input must be a grayscale image (2D numpy array).")
    k = _odd(max(3, int(0.03 * min(gray.shape[:2]))))
    bg = cv2.medianBlur(gray, k)
    return cv2.divide(gray, bg, scale=128)

def _apply_clahe(gray: np.ndarray, clip=2.0, tile=8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Args:
        gray (np.ndarray): Input grayscale image.
        clip (float): CLAHE clip limit.
        tile (int): CLAHE tile grid size.

    Returns:
        np.ndarray: Contrast-enhanced image.

    Raises:
        ValueError: If input is not a valid grayscale image.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input must be a grayscale image (2D numpy array).")
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    return clahe.apply(gray)

def _auto_gamma_for_bg(gray: np.ndarray, target_bg: int = 245) -> np.ndarray:
    """
    Adjust the gamma of an image to normalize the background intensity.

    Args:
        gray (np.ndarray): Input grayscale image.
        target_bg (int): Target background intensity (0-255).

    Returns:
        np.ndarray: Gamma-adjusted image.

    Raises:
        ValueError: If input is not a valid grayscale image.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input must be a grayscale image (2D numpy array).")
    med = float(np.median(gray))
    if med <= 1 or med >= 254:
        return gray
    t = target_bg / 255.0
    m = med / 255.0
    gamma = np.log(t + 1e-6) / np.log(m + 1e-6)
    gamma = np.clip(gamma, 0.5, 2.5)
    lut = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255.0).astype(np.uint8)
    return cv2.LUT(gray, lut)

def _denoise_bilateral(gray: np.ndarray) -> np.ndarray:
    """
    Denoise an image using bilateral filtering.

    Args:
        gray (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Denoised image.

    Raises:
        ValueError: If input is not a valid grayscale image.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input must be a grayscale image (2D numpy array).")
    return cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=20)

def _unsharp(gray: np.ndarray, sigma=1.0, amount=0.7) -> np.ndarray:
    """
    Apply unsharp masking to enhance image sharpness.

    Args:
        gray (np.ndarray): Input grayscale image.
        sigma (float): Standard deviation for Gaussian blur in unsharp masking.
        amount (float): Amount of sharpening.

    Returns:
        np.ndarray: Sharpened image.

    Raises:
        ValueError: If input is not a valid grayscale image.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError("Input must be a grayscale image (2D numpy array).")
    if amount <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0,0), sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def _smart_grayscale(bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale using the best candidate channel
    based on Otsu's method for separability.

    Args:
        bgr (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: Grayscale image.

    Raises:
        ValueError: If input is not a valid BGR image.
    """
    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Input must be a BGR image (3-channel numpy array).")
    g_y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    b, g, r = cv2.split(bgr)
    min_rgb = cv2.min(cv2.min(b, g), r)

    def _otsu_score(img8):
        hist = cv2.calcHist([img8],[0],None,[256],[0,256]).ravel()
        p = hist / (img8.size + 1e-9)
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1 - omega) + 1e-9)
        return float(np.nanmax(sigma_b2))

    cands = [(g_y, _otsu_score(g_y)), (v, _otsu_score(v)), (min_rgb, _otsu_score(min_rgb))]
    return max(cands, key=lambda t: t[1])[0]

def convert_to_bw(
    img: np.ndarray,
    method: str = "safe_for_ocr",
    threshold: int = 50,
    max_value: int = 255,
    adaptive_block_size: int = 51,
    adaptive_C: int = 2,
    use_clahe: bool = False,
    suppress_highlighter: bool = True,
    invert: bool = False,
    try_sauvola: bool = True
) -> np.ndarray:
    """
    Convert an image to a binarized or enhanced grayscale version for OCR/document analysis.

    Args:
        img (np.ndarray): Input image (BGR or grayscale).
        method (str): Binarization method: 'safe_for_ocr', 'threshold', 'otsu', 'adaptive', 'blackhat', or 'sauvola'.
        threshold (int): Threshold value for simple thresholding.
        max_value (int): Maximum value for thresholding.
        adaptive_block_size (int): Block size for adaptive methods.
        adaptive_C (int): Constant subtracted from mean in adaptive methods.
        use_clahe (bool): Whether to apply CLAHE contrast enhancement.
        suppress_highlighter (bool): Whether to suppress highlighter marks.
        invert (bool): Whether to invert the output image.
        try_sauvola (bool): Whether to use Sauvola binarization if available.

    Returns:
        np.ndarray: Binarized or enhanced grayscale image.

    Raises:
        ValueError: If an unsupported method is specified or input is invalid.

    Usage example:
        >>> bw = convert_to_bw(image, method="otsu")
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    """
    if method == "safe_for_ocr":
        # return basic grayscale for OCR - no enchancements
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return gray
    """
    
    if img.ndim == 2:
        gray0 = img.copy()
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img
        if suppress_highlighter:
            bgr = suppress_highlighter_hsv(bgr)
        gray0 = _smart_grayscale(bgr)
        
    if method == "safe_for_ocr":
        return gray0

    #gray = _deskew(gray0)  # Deskewing can be enabled if needed
    gray = _bg_normalize_median(gray0)
    gray = _bg_normalize(gray, sigma=35.0)

    if use_clahe:
        gray = _apply_clahe(gray, clip=2.0, tile=8)
    gray = _auto_gamma_for_bg(gray, target_bg=245)
    gray = _denoise_bilateral(gray)
    gray = _unsharp(gray, sigma=1.0, amount=0.7)      

    thresh_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if method == "threshold":
        _, bw = cv2.threshold(gray, threshold, max_value, thresh_flag)
    elif method == "otsu":
        _, bw = cv2.threshold(gray, 0, max_value, thresh_flag + cv2.THRESH_OTSU)
    elif method == "adaptive":
        b = _odd(adaptive_block_size)
        gray_blur = cv2.GaussianBlur(gray, (3,3), 0)
        bw = cv2.adaptiveThreshold(gray_blur, max_value,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   thresh_flag, b, adaptive_C)
        bw = _remove_speckles(bw, min_area=12)
    elif method == "sauvola":
        if try_sauvola and hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "niBlackThreshold"):
            g = cv2.GaussianBlur(gray, (3,3), 0)
            k = 0.5
            bw = cv2.ximgproc.niBlackThreshold(g, max_value,
                                               cv2.THRESH_BINARY if not invert else cv2.THRESH_BINARY_INV,
                                               blockSize=_odd(adaptive_block_size),
                                               k=k,
                                               binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
            bw = _remove_speckles(bw, min_area=12)
        else:
            b = _odd(adaptive_block_size)
            g = cv2.GaussianBlur(gray, (3,3), 0)
            bw = cv2.adaptiveThreshold(g, max_value,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresh_flag, b, adaptive_C)
            bw = _remove_speckles(bw, min_area=12)
    elif method == "blackhat":
        blur = cv2.bilateralFilter(gray, d=7, sigmaColor=20, sigmaSpace=20)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        close = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        bh = cv2.subtract(close, blur)
        bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
        _, bw = cv2.threshold(bh, 0, max_value, thresh_flag + cv2.THRESH_OTSU)
        if invert:
            bw = 255 - bw
        bw = _remove_speckles(bw, min_area=12)
    else:
        raise ValueError("Unsupported method. Use 'safe_for_ocr', 'threshold', 'otsu', 'adaptive', 'blackhat', or 'sauvola'.")

    return bw

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by a given angle (degrees).

    Args:
        img (np.ndarray): Input image (grayscale or BGR).
        angle (float): Angle in degrees.

    Returns:
        np.ndarray: Rotated image.

    Raises:
        ValueError: If input is not a valid image.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if len(img.shape) == 2:
        (h, w) = img.shape
    else:
        (h, w, _) = img.shape
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return rotated

def detect_lines(
    img: np.ndarray,
    min_horz_length: int = 100,
    min_vert_length: int = 100,
    overlay: bool = True,
    horizontal_color: Tuple[int, int, int] = (0, 0, 255),
    vertical_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Detect long horizontal/vertical lines and optionally overlay them on the image.

    Args:
        img (np.ndarray): Input image (grayscale or BGR).
        min_horz_length (int): Minimum horizontal line length.
        min_vert_length (int): Minimum vertical line length.
        overlay (bool): If True, overlay lines with color; else return mask.
        horizontal_color (tuple): Color for horizontal lines (BGR).
        vertical_color (tuple): Color for vertical lines (BGR).

    Returns:
        np.ndarray: Image with detected lines overlayed or mask.

    Raises:
        ValueError: If input is not a valid image.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (min_horz_length, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=1)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vert_length))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=1)
    if overlay:
        overlay_img = colored.copy()
        overlay_img[horizontal > 0] = horizontal_color
        overlay_img[vertical > 0] = vertical_color
        return overlay_img
    combined = cv2.bitwise_or(horizontal, vertical)
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

def _preprocess_image_bytes(
    image_bytes: bytes,
    img_format: str,
    threshold: int,
    method: str,
    angle: Optional[float],
    hide_h_line: bool,
    hide_v_lines: bool,
) -> bytes:
    """
    Preprocess a single image page from bytes.

    Args:
        image_bytes (bytes): Image data in bytes.
        img_format (str): Output image format.
        threshold (int): Threshold value for binarization.
        method (str): Binarization method.
        angle (float, optional): Angle for rotation.
        hide_h_line (bool): Whether to hide horizontal lines.
        hide_v_lines (bool): Whether to hide vertical lines.

    Returns:
        bytes: Preprocessed image as bytes.

    Raises:
        Exception: If preprocessing fails.
    """
    try:
        img: np.ndarray = open_image_as(image_bytes, output_format="cv")
        img = convert_to_bw(img, threshold=threshold, method=method)
        if angle not in (None, 0):
            img = rotate_image(img, angle)
        if hide_h_line or hide_v_lines:
            vertical_color = (255, 255, 255) if hide_v_lines else (0, 0, 0)
            horizontal_color = (255, 255, 255) if hide_h_line else (0, 0, 0)
            img = detect_lines(
                img,
                overlay=True,
                vertical_color=vertical_color,
                horizontal_color=horizontal_color,
            )
        return image_to_bytes(img, img_format)
    except Exception as e:
        raise Exception(f"Failed to preprocess image bytes: {e}") from e

def do_images_preprocesing(
    raw_images: List[bytes],
    img_format: str = "PNG",
    method: str = "safe_for_ocr",
    threshold: int = 200,
    angle: Optional[float] = None,
    hide_h_line: bool = False,
    hide_v_lines: bool = False,
) -> List[bytes]:
    """
    Preprocess a list of image bytes in parallel.

    Args:
        raw_images (List[bytes]): List of image bytes.
        img_format (str): Output image format.
        method (str): Binarization method.
        threshold (int): Threshold value for binarization.
        angle (float, optional): Angle for rotation.
        hide_h_line (bool): Whether to hide horizontal lines.
        hide_v_lines (bool): Whether to hide vertical lines.

    Returns:
        List[bytes]: List of preprocessed image bytes.

    Raises:
        Exception: If preprocessing fails.
    """
    try:
        max_workers = min(32, (os.cpu_count() or 1) * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            outputs: List[bytes] = list(
                ex.map(
                    _preprocess_image_bytes,
                    raw_images,
                    repeat(img_format),
                    repeat(threshold),
                    repeat(method),
                    repeat(angle),
                    repeat(hide_h_line),
                    repeat(hide_v_lines),
                )
            )
        return outputs
    except Exception as e:
        raise Exception(f"Failed to preprocess images: {e}") from e

def documents_to_images_conversion(
    file: str | bytes,
    pdf_convertor_endpoint: str,
    img_format: str = "PNG"
) -> List[bytes]:
    """
    Convert a document (PDF or image) to a list of image bytes (one per page).

    Args:
        file (str | bytes): File path or file bytes.
        pdf_convertor_endpoint (str): Endpoint for PDF/image conversion.
        img_format (str): Output image format.

    Returns:
        List[bytes]: List of image bytes.

    Raises:
        ValueError: If input type is not supported.
        Exception: If conversion fails.
    """
    try:
        if isinstance(file, str):
            binary_data = convert_to_binary(file)
        elif isinstance(file, bytes):
            binary_data = file
        else:
            raise ValueError(f"Unsupported input type: {type(file)}")
        mime_type = MimeDetector.detect_mime_type(binary_data)
        # print(f"[DEBUG] Detected mime_type: {mime_type}")
        raw_images: List[bytes] = Convertors.convert_to_images(
            binary_data,
            document_image_render_format=img_format,
            pdf_convertor_endpoint=pdf_convertor_endpoint,
            mime_type=mime_type,
        )
        return raw_images
    except Exception as e:
        raise Exception(f"Failed to convert document to images: {e}") from e