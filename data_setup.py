from PIL import Image
import config
from pathlib import Path


def load_image_pil(image_path: Path):
    """
    Loads an image using PIL from the given path.
    Returns a PIL Image object in RGB format, or None if an error occurs.
    """
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return None
    try:
        img = Image.open(image_path).convert('RGB')
        print(f"Image loaded successfully from: {image_path}")
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


if __name__ == '__main__':
    # Example usage:
    test_image_path = config.INPUT_IMAGE_DIR / config.DEFAULT_IMAGE_NAME
    if not test_image_path.exists():
        print(
            f"Please place an image named '{config.DEFAULT_IMAGE_NAME}' in '{config.INPUT_IMAGE_DIR}' for testing.")
    else:
        pil_image = load_image_pil(test_image_path)
        if pil_image:
            print(f"Loaded image dimensions: {pil_image.size}")
            # pil_image.show() # Uncomment to display
