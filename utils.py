import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import config


def draw_detections(pil_image: Image.Image, boxes, probs, landmarks):
    """
    Draws bounding boxes, probabilities, and landmarks on a copy of the PIL image.
    Returns an OpenCV image (BGR format) with detections drawn.
    """
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    if boxes is not None:
        for i, (box, prob, landmark_points) in enumerate(zip(boxes, probs, landmarks)):
            if prob < config.CONFIDENCE_THRESHOLD_DISPLAY:  # Filter by confidence
                continue

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)  # Green box

            # Put probability text
            label = f'{prob:.2f}'
            cv2.putText(img_cv, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw landmarks
            if landmark_points is not None:
                for point in landmark_points:
                    x, y = map(int, point)
                    # Red circles
                    cv2.circle(img_cv, (x, y), 3, (0, 0, 255), -1)
    return img_cv


def display_image_cv(cv_image, window_name="Face Detections"):
    """Displays an OpenCV image."""
    cv2.imshow(window_name, cv_image)
    print(
        f"Displaying image in window: '{window_name}'. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image_cv(cv_image, output_path: Path):
    """Saves an OpenCV image to the specified path."""
    try:
        cv2.imwrite(str(output_path), cv_image)
        print(f"Image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")


def generate_output_filename(input_filename: str, suffix="_detected"):
    """Generates an output filename with a timestamp and suffix."""
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    base, ext = Path(input_filename).stem, Path(input_filename).suffix
    return f"{base}{suffix}_{timestamp_str}{ext}"


if __name__ == '__main__':
    # Example: Create a dummy PIL image and draw on it
    dummy_pil_image = Image.new('RGB', (600, 400), color='lightblue')
    # Dummy detections
    dummy_boxes = np.array([[100, 100, 300, 300], [350, 50, 450, 150]])
    dummy_probs = np.array([0.99, 0.95])
    dummy_landmarks = np.array([
        [[120, 150], [280, 150], [200, 200], [150, 250],
            [250, 250]],  # Landmarks for face 1
        [[370, 70], [430, 70], [400, 100], [380, 120],
            [420, 120]]   # Landmarks for face 2
    ])

    img_with_detections = draw_detections(
        dummy_pil_image, dummy_boxes, dummy_probs, dummy_landmarks)
    # display_image_cv(img_with_detections, "Utils Test")

    output_fname = generate_output_filename("result_image.jpg")
    print(f"Generated output filename: {output_fname}")
    # save_image_cv(img_with_detections, config.OUTPUT_IMAGE_DIR /
    #               output_fname)  # Test saving
