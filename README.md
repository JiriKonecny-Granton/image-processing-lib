# image-preprocessing-lib

A Python library for advanced image preprocessing and visualization, designed for document analysis and OCR workflows.

## Features

- **Highlighter suppression** in color images
- **Deskewing** of scanned documents
- **Background normalization** (Gaussian and median)
- **Smart grayscale conversion**
- **Multiple binarization methods**: threshold, Otsu, adaptive, Sauvola, blackhat
- **Noise and speckle removal**
- **Batch and parallel preprocessing** of multi-page documents
- **Line detection and removal**
- **Integration with external convertors** for PDF/image conversion
- **Visualization utilities** for Jupyter Notebooks

## Installation

```bash
pip install .
```

Or add to your `pyproject.toml`:

```toml
[project.dependencies]
image-preprocessing-lib = "*"
```

## Dependencies

- numpy >= 1.22
- opencv-python >= 4.5
- Pillow >= 9.0
- GrtUtils
- MimeDetector
- ipython >= 7.0

## Usage

### Preprocessing a Document

```python
from preprocessing import convert_and_preprocess_document

# Preprocess a PDF or image file
converted_pages = documents_to_images_conversion("document.pdf", <PDF_CONVERSION_ENDPOINT>)
processed_pages = do_images_preprocesing(converted_pages, method="otsu")
# processed_pages is a list of bytes, each representing a preprocessed image page
```

### Visualizing Images in Jupyter

```python
from visualise import show_as_image

show_as_image(processed_pages[0], title="First Page", w=600)
```

## API Overview

- `documents_to_images_conversion(document, pdf_conversion_endpoint:str)`  
  Convert document to a list of image bytes in parallel.

- `do_images_preprocesing(raw_images, method="threshold", ...)`  
  Preprocess a list of image bytes in parallel.

- `combine_pages_vertically(pages_bytes)`  
  Combine multiple image pages vertically into a single image.

- `show_as_image(file, title=None, w=None, h=None)`  
  Display an image or list of images in a Jupyter Notebook.

## License

This is private property of Granton company.

## Author

Granton s.r.o.  
[info@granton.cz](mailto:info@granton.cz)

## Homepage

[https://github.com/granton/image-preprocessing-lib](https://github.com/granton/image-preprocessing-lib)