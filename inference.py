# inference.py
import torch
import argparse
from pathlib import Path

import config
from data_setup import load_image_pil
from model_loader import get_face_detector
from utils import draw_detections, save_image_cv, generate_output_filename


def main(image_path_str: str):
    """
    Performs face detection on a single image.
    """
    print(f"Starting face detection inference on device: {config.DEVICE}")

    image_path = Path(image_path_str)
    if not image_path.is_file():
        # Try to find it in the default input directory
        image_path = config.INPUT_IMAGE_DIR / image_path.name
        if not image_path.is_file():
            print(
                f"Error: Image not found at '{image_path_str}' or in '{config.INPUT_IMAGE_DIR}'.")
            # Try using the default image if specified path is invalid
            print(
                f"Attempting to use default image: {config.DEFAULT_IMAGE_NAME}")
            image_path = config.INPUT_IMAGE_DIR / config.DEFAULT_IMAGE_NAME
            if not image_path.is_file():
                print(
                    f"Error: Default image '{config.DEFAULT_IMAGE_NAME}' also not found in '{config.INPUT_IMAGE_DIR}'.")
                print(
                    "Please provide a valid image path or place an image in the input directory.")
                return

    # 1. Load image
    pil_image = load_image_pil(image_path)
    if pil_image is None:
        return

    # 2. Load model (face detector)
    detector = get_face_detector()

    # 3. Perform detection
    # MTCNN.detect() returns:
    # boxes: list of [x1, y1, x2, y2]
    # probs: list of probabilities
    # landmarks: list of [ [x_le, y_le], [x_re, y_re], ..., [x_rm, y_rm] ]
    # The inputs to detect should be PIL Image or numpy array or torch tensor.
    # No need to manually move PIL image to device, MTCNN handles it.
    print(f"\nPerforming detection on: {image_path.name}...")
    try:
        with torch.no_grad():  # Inference mode
            boxes, probs, landmarks = detector.detect(
                pil_image, landmarks=True)
    except Exception as e:
        print(f"Error during detection: {e}")
        # This can happen with very small images or if CUDA runs out of memory
        return

    # 4. Process and display/save results
    if boxes is not None:
        print(f"Detected {len(boxes)} face(s).")
        # Filter by confidence (already handled in draw_detections, but good to know)
        # valid_indices = [i for i, p in enumerate(probs) if p >= config.CONFIDENCE_THRESHOLD_DISPLAY]
        # boxes = boxes[valid_indices]
        # probs = probs[valid_indices]
        # landmarks = landmarks[valid_indices]
        # print(f"Displaying {len(boxes)} face(s) with confidence >= {config.CONFIDENCE_THRESHOLD_DISPLAY}.")

        img_with_detections = draw_detections(
            pil_image, boxes, probs, landmarks)

        # display_image_cv(img_with_detections,
        #                  f"Detections on {image_path.name}")

        output_filename = generate_output_filename(image_path.name)
        output_save_path = config.OUTPUT_IMAGE_DIR / output_filename
        save_image_cv(img_with_detections, output_save_path)
    else:
        print("No faces detected in the image.")
        # Display original image if no detections
        # display_image_cv(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), f"No Detections - {image_path.name}")

    print("\nInference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection using MTCNN")
    parser.add_argument(
        "--image",
        type=str,
        # Use default from config
        default=str(config.INPUT_IMAGE_DIR / config.DEFAULT_IMAGE_NAME),
        help="Path to the input image file."
    )
    args = parser.parse_args()

    main(args.image)
