import numpy as np
import cv2


def crop_image(image, point1, point2, margin=(0, 0)):
    """
    Crop the image based on two coordinates (top-left and bottom-right).
    Args:
        image (array): Original image.
        top_left (tuple): Top-left coordinate (x, y) of the cropping area.
        bottom_right (tuple): Bottom-right coordinate (x, y) of the cropping area.
    Returns:
        array: Cropped image.
    """
    # Calculate bounding box coordinates with margin
    x1 = max(0, point1[0] - margin[0])
    y1 = max(0, point1[1] - margin[1])
    x2 = min(image.shape[1], point2[0] + margin[0])
    y2 = min(image.shape[0], point2[1] + margin[1])
    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


def get_document_rectangle(point1, point2, margin=(5, 5)):
    """
    Given two points, return the coordinates of the rectangle [x1, y1, x2, y2]
    with an optional margin.
    """
    # Extract coordinates from points
    x1, y1 = point1
    x2, y2 = point2

    # Apply the margin
    x1 -= margin[0]
    y1 -= margin[1]
    x2 += margin[0]
    y2 += margin[1]

    # Ensure coordinates are ordered correctly
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    return (x1, y1, x2, y2)


def update_box_coordination(start_points, boxes):
    crop_x, crop_y = start_points
    updated_boxes = [
        [[point[0] + crop_x, point[1] + crop_y] for point in box] for box in boxes
    ]
    return updated_boxes
