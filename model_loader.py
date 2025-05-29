import torch
from facenet_pytorch import MTCNN
import config


def get_face_detector():
    """
    Initializes and returns a pre-trained MTCNN face detector.
    """
    print(f"Initializing MTCNN detector on device: {config.DEVICE}")
    mtcnn = MTCNN(
        # Not strictly for detection, but can influence internal processing
        image_size=config.MTCNN_IMAGE_SIZE,
        margin=config.MTCNN_MARGIN,           # Not strictly for detection
        min_face_size=config.MTCNN_MIN_FACE_SIZE,
        thresholds=config.MTCNN_THRESHOLDS,
        factor=config.MTCNN_FACTOR,
        post_process=config.MTCNN_POST_PROCESS,
        keep_all=config.MTCNN_KEEP_ALL,
        device=config.DEVICE
    )

    print("MTCNN detector initialized.")
    return mtcnn


if __name__ == '__main__':
    # Example usage:
    detector = get_face_detector()
    print(f"Detector type: {type(detector)}")
    # This just confirms the model loads. Actual detection is in inference.py
