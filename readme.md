# PhotoCropper an Image Processing Script

Here’s a description for the `process_images` script for your README file:

---

### Image Processing Script (`process_images.py`)

The `process_images.py` script is designed to detect faces in images, expand the detected bounding boxes around the
faces to include more context (such as head and shoulders), and then crop and resize the images to a specified aspect
ratio. This script is ideal for preparing images for tasks where consistent framing of faces is needed, such as in
facial recognition, dataset generation, or portrait adjustments.

#### Key Features:

1. **Face Detection:**
    - Uses a deep learning model (e.g., YOLOv8 or YOLOv3) to detect faces within each image. Detected face bounding
      boxes are automatically generated.

2. **Bounding Box Expansion:**
    - The detected bounding box is expanded to capture more of the head, shoulders, and surrounding context.
    - The expansion is adaptive and depends on the face size in relation to the image size:
        - Small faces get larger expansions.
        - Large faces get minimal expansions.
    - The expansion respects the aspect ratio to ensure that the final crop has consistent dimensions.

3. **Aspect Ratio Correction:**
    - The script ensures that the final cropped region matches the desired aspect ratio (e.g., 896x1152).
    - If necessary, the crop is adjusted to maintain the aspect ratio, either by adjusting the width or height.

4. **Padding and Resizing:**
    - If the expanded bounding box results in a crop that’s smaller than the desired output size, the image is padded
      with a black background to meet the required dimensions.
    - Finally, the cropped and/or padded image is resized to the target dimensions using linear interpolation.

5. **Debugging and Visualization:**
    - For debugging purposes, the script saves a copy of the original image with visual markers indicating:
        - The detected face bounding box.
        - The expanded bounding box used for cropping.
        - The final crop used for resizing.
    - Each step of the process is clearly visualized, aiding in the understanding of how the bounding box was
      transformed.

6. **Output:**
    - The final processed images are saved in the specified directory with the desired resolution and aspect ratio.
    - Optionally, debug images showing bounding box transformations are also saved in the `debug_directory`.

## Requirements

- Python 3.10
- Any additional dependencies required by `main_v5.py`
- yolov8n-face.pt

## Installation

1. Clone the repository or download the script.
2. Download yolov8n-face.pt and put in root directory.
3. Install the required dependencies for `main_v5.py` (if not already installed):

   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run the script from the command line using the following format:

```bash
python process_images.py --input_dir <input_directory> --output_dir <output_directory> [--debug_dir <debug_directory>]
```

### Arguments

- `--input_dir`: **(Required)** Path to the input directory containing the images to be processed.
- `--output_dir`: **(Required)** Path to the output directory where processed images will be saved.
- `--debug_dir`: **(Optional)** Path to the directory for saving debug files. If not provided, debug files will not be
  generated.

### Example

```bash
python process_images.py --input_dir "J:/AI/Train/Regularization/TBD" --output_dir "J:/AI/Train/Regularization/TBD/processed" --debug_dir "J:/AI/Train/Regularization/TBD/debug"
```

In this example:

- Images will be processed from `J:/AI/Train/Regularization/TBD`.
- Processed images will be saved in `J:/AI/Train/Regularization/TBD/processed`.
- Debug files will be saved in `J:/AI/Train/Regularization/TBD/debug`.

### Note

Ensure that the directories provided in the arguments exist, and you have appropriate read/write permissions.
