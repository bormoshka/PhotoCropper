import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO
import math

model = YOLO('yolov8n-face.pt')


def draw_debug_info(image, face_box, expand_box, box_after_respect, crop_box, output_dir, filename):
    debug_img = image.copy()

    # Draw and label the face bounding box
    x, y, w, h = face_box
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.putText(debug_img, 'Face Box', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Draw and label the expanded bounding box
    x1, y1, x2, y2 = expand_box
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (125, 26, 255), 2)
    cv2.putText(debug_img, 'Expanded Box', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (125, 26, 255), 2)

    # Draw and label the box before respecting aspect ratio
    # x1, y1, x2, y2 = box_before_respect
    # cv2.rectangle(debug_img, (x1, y1), (x2, y2), (44, 26, 128), 2)
    # cv2.putText(debug_img, 'Box Before Respect', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (44, 26, 128), 2)

    # Draw and label the box after respecting aspect ratio
    x1, y1, x2, y2 = box_after_respect
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (44, 26, 128), 2, lineType=cv2.FILLED)
    cv2.putText(debug_img, 'Box After Respect', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (44, 26, 128), 2)

    # Draw and label the final crop bounding box
    x1, y1, x2, y2 = crop_box
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(debug_img, 'Crop Box', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Save the debug image
    cv2.imwrite(f'{output_dir}/{filename}', debug_img)


def detect_faces_yolov8(image):
    """
    Detect faces in an image using YOLOv8.

    Parameters:
        image (np.ndarray): The input image in which faces are to be detected.

    Returns:
        List[dict]: List of dictionaries containing bounding boxes and confidence scores for detected faces.
    """
    # Convert image from BGR to RGB (YOLO models expect RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(rgb_image)

    faces = []

    # Iterate over the detections
    for result in results:
        for detection in result.boxes:
            # Extract the bounding box and confidence
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            confidence = detection.conf[0]

            # Filter out weak detections
            if confidence > 0.5:
                faces.append({
                    'box': (x1, y1, x2 - x1, y2 - y1),  # (x, y, width, height)
                    'confidence': confidence
                })

    return faces


def expand_bounding_box(x, y, w, h, img_width, img_height, w_scale_factor=1.5, h_scale_factor=1.5,
                        desired_aspect_ratio=896 / 1152):
    """
       Scale the bounding box, with given custom while preserving the aspect ratio and ensuring it remains within the image boundaries.
       Parameters:
           x (int): The left-most x coordinate of the bounding box.
           y (int): The top-most y coordinate of the bounding box.
           w (int): The width of the bounding box.
           h (int): The height of the bounding box.
           img_width (int): The image's total width.
           img_height (int): The image's total height.
           w_scale_factor (float, optional=1.5): Scale factor for the width.
           h_scale_factor (float, optional=1.5): Scale factor for the height.
           desired_aspect_ratio (float, optional=896 / 1152): Desired aspect ratio of the resulting bounding box.
       Returns:
           int, int, int, int: The new x and y coordinates, width, and height of the expanded bounding box.
       """
    aspect_ratio = w / h

    if aspect_ratio < desired_aspect_ratio:
        # The width is the smaller dimension, so we'll expand it
        scaled_width = int(w * w_scale_factor)
        scaled_height = int(scaled_width / desired_aspect_ratio)
    else:
        # The height is the smaller dimension, so we'll expand it
        scaled_height = int(h * h_scale_factor)
        scaled_width = int(scaled_height * desired_aspect_ratio)

    # Center of the original bounding box
    cx = x + w / 2
    cy = y + h / 2

    # Ensure the new bounding box fits within the image boundaries
    left = max(0, cx - scaled_width // 2)
    top = max(0, cy - scaled_height // 2)
    right = min(img_width, left + scaled_width)
    bottom = min(img_height, top + scaled_height)

    print("left, top, right, bottom", left, top, right, bottom)

    # Adjust the left/top if the new dimensions exceed the image boundaries
    if right > img_width:
        right = img_width
        left = right - scaled_width

    if bottom > img_height:
        bottom = img_height
        top = bottom - scaled_height

    left, top, width, height = int(round(left)), int(round(top)), int(round(right - left)), int(round(bottom - top))
    aspect_ratio = width / height

    if aspect_ratio < desired_aspect_ratio:
        height = int(width / desired_aspect_ratio)
    else:
        width = int(height * desired_aspect_ratio)
    # Return the adjusted bounding box coordinates
    return left, top, width, height


def calculate_scale_factors(face_box, img_width, img_height, mode="Portrait", aspect_ratio=None):
    """
    Calculate scale factors for expanding the bounding box based on the size of the face
    relative to the entire image and desired orientation/aspect ratio.

    Parameters:
    - face_box: Tuple of (x, y, width, height) representing the face bounding box.
    - img_width: Width of the image.
    - img_height: Height of the image.
    - mode: String, either "Portrait" or "Landscape" to specify the orientation.
    - aspect_ratio: Optional float to specify the desired aspect ratio.
                    If None, default aspect ratio for mode will be used.

    Returns:
    - w_scale_factor: Scaling factor for width.
    - h_scale_factor: Scaling factor for height.
    """
    # Calculate the area of the face bounding box
    face_x, face_y, face_w, face_h = face_box
    face_area = face_w * face_h

    # Calculate the area of the entire image
    image_area = img_width * img_height

    # Calculate the proportion of the face area to the image area
    face_proportion = face_area / image_area
    min_scale_factor = 0.85
    max_scale_factor = 2.0
    print(f"face_proportion: {face_proportion}, face_w: {face_w}, face_h: {face_h} = face_area: {face_area}, img_width: {img_width}, img_height: {img_height}, image_area: {image_area}")
    # Introduce a non-linear dependency using the square root
    scaling_factor = (0.8 / (math.sqrt(face_proportion) + 0.01) / 1.1) - 1.0
    if scaling_factor > max_scale_factor:
        scaling_factor = max_scale_factor
    if scaling_factor < min_scale_factor:
        scaling_factor = min_scale_factor

    # Default aspect ratios based on mode
    if aspect_ratio is None:
        aspect_ratio = 896 / 1152 if mode == "Portrait" else 1152 / 896

    # Define scale factor ranges
    max_scale_factor_long_side = 2.5  # Max scaling when face is very small
    max_scale_factor_short_side = 2.0  # Max scaling when face is very small
    min_scale_factor_short_side = 0.5  # Min scaling when face is very large
    min_scale_factor_long_side = 0.8  # Min scaling when face is very large
    # Adjust scale factors to match the aspect ratio
    if mode == "Portrait":
        # Adjust the scaling factors using the non-linear dependency
        w_scale_factor = max_scale_factor_short_side * scaling_factor
        h_scale_factor = max_scale_factor_short_side * scaling_factor

        if w_scale_factor / h_scale_factor > aspect_ratio:
            w_scale_factor = h_scale_factor * aspect_ratio
        else:
            h_scale_factor = w_scale_factor / aspect_ratio
    elif mode == "Landscape":
        # Adjust the scaling factors using the non-linear dependency
        w_scale_factor = max_scale_factor_long_side * scaling_factor
        h_scale_factor = min_scale_factor_short_side * scaling_factor
        if h_scale_factor / w_scale_factor > aspect_ratio:
            h_scale_factor = w_scale_factor * aspect_ratio
        else:
            w_scale_factor = h_scale_factor / aspect_ratio

    return max(1, w_scale_factor), max(1, h_scale_factor)


def crop_and_resize(image, filename, debug_directory, face_box, desired_size=(896, 1152)):
    print(face_box)
    """
    Crop the image around the expanded bounding box and resize to the desired size.
    """
    img_height, img_width = image.shape[:2]
    desired_width, desired_height = desired_size
    desired_aspect = desired_width / desired_height
    s_w, s_h = calculate_scale_factors(face_box, img_width, img_height)

    # Expand the face bounding box to include more of the head and shoulders
    x, y, w, h = expand_bounding_box(*face_box, img_width, img_height, s_w, s_h, desired_aspect_ratio=desired_aspect)
    print(f"x: {x}, y: {y}, width: {w}, height: {h}, s_w: {s_w}, s_h: {s_h}")

    expanded_bounding_box = (x, y, x + w, y + h)
    # Calculate the center of the expanded bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    crop_width = w
    crop_height = h

    print(f"crop_width: {crop_width}, crop_height: {crop_height}")

    # CORNER CASE. Ensure the crop box is large enough to fit the desired resolution
    if crop_width < desired_width:
        crop_width = desired_width
        crop_height = int(desired_width / desired_aspect)
    elif crop_height < desired_height:
        crop_height = desired_height
        crop_width = int(desired_height * desired_aspect)

    # Calculate the new top-left corner of the crop box
    crop_x1 = max(0, int(center_x - crop_width // 2))
    crop_y1 = max(0, int(center_y - crop_height // 2))

    # Adjust the crop box if it goes out of image bounds
    crop_x2 = min(img_width, crop_x1 + crop_width)
    crop_y2 = min(img_height, crop_y1 + crop_height)

    # Ensure the crop box respects the image boundaries
    print("Before Ensure", crop_x1, crop_y1, crop_x2, crop_y2)
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img_width, crop_x2)
    crop_y2 = min(img_height, crop_y2)
    box_after_respect = (crop_x1, crop_y1, crop_x2, crop_y2)
    print("after Ensure", crop_x1, crop_y1, crop_x2, crop_y2)

    # Check if the crop box is large enough, and adjust if needed
    if (crop_x2 - crop_x1) < desired_width or (crop_y2 - crop_y1) < desired_height:
        scale_w = desired_width / (crop_x2 - crop_x1)
        scale_h = desired_height / (crop_y2 - crop_y1)
        scale = max(scale_w, scale_h)

        new_crop_width = int(crop_width * scale)
        new_crop_height = int(crop_height * scale)

        crop_x1 = max(0, int(center_x - new_crop_width // 2))
        crop_y1 = max(0, int(center_y - new_crop_height // 2))
        crop_x2 = min(img_width, crop_x1 + new_crop_width)
        crop_y2 = min(img_height, crop_y1 + new_crop_height)

        print("after check", crop_x1, crop_y1, crop_x2, crop_y2)

    # Crop the image
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Calculate the size of the cropped image
    cropped_height, cropped_width = cropped_image.shape[:2]

    # Create a new image with the desired size and a black background
    if cropped_width < desired_width or cropped_height < desired_height:
        padded_image = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)
        x_offset = (desired_width - cropped_width) // 2
        y_offset = (desired_height - cropped_height) // 2
        padded_image[y_offset:y_offset + cropped_height, x_offset:x_offset + cropped_width] = cropped_image
    else:
        padded_image = cropped_image

    # Resize the image to the desired size
    resized_image = cv2.resize(padded_image, desired_size, interpolation=cv2.INTER_LINEAR)

    # Save debug information if output_dir and filename are provided
    if debug_directory and filename:
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
        draw_debug_info(image, face_box, expanded_bounding_box, box_after_respect, crop_box, debug_directory, filename)

    return resized_image


def process_images(input_dir, output_dir, debug_directory, desired_size=(896, 1152)):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Failed to load {filename}. Skipping.")
                continue

            # Detect faces using MTCNN
            results = detect_faces_yolov8(image)

            if len(results) > 0:
                # Select the largest face found
                face = max(results, key=lambda rect: rect['box'][2] * rect['box'][3])
                face_box = face['box']
                # If the face is of reasonable size, process the image
                processed_image = crop_and_resize(image, filename, debug_directory, face_box=face_box,
                                                  desired_size=desired_size)

                # Save the processed image to the output directory
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, processed_image)
            else:
                # If no face is detected, save the image with "failed_" prefix
                failed_output_path = os.path.join(output_dir, f"failed_{filename}")
                shutil.copyfile(img_path, failed_output_path)
                print(f"Face detection failed, saved unmodified image as: {failed_output_path}")
