import argparse
from main_v5 import process_images


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process images from input directory and save the results to output directory.")

    # Define arguments
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the input directory containing the images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory to save the processed images.')
    parser.add_argument('--debug_dir', type=str, required=False, default=None,
                        help='Path to the debug directory for saving debug files (optional).')

    # Parse arguments
    args = parser.parse_args()

    # Call the process_images function with the provided arguments
    process_images(args.input_dir, args.output_dir, args.debug_dir)


if __name__ == "__main__":
    main()