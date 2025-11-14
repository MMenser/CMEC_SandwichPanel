import os
from PIL import Image

INPUT_FOLDER = r'C:\Users\mason\Work\CMEC_SandwichPanel\Images\Images'
OUTPUT_FOLDER = r'C:\Users\mason\Work\CMEC_SandwichPanel\Images\Processed_128x128_Grayscale'
TARGET_SIZE = (128, 128)

def convert_to_greyscale_and_resize(input_dir, output_dir, size):
    if not os.path.isdir(input_dir):
        print(f"Error: Input folder '{input_dir}' not found.")
        return
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory ensured: '{output_dir}'")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    print(f"Starting conversion from '{input_dir}'...")

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'): # Process only .png files
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename) # New path for saving

            try:
                # Open the image
                with Image.open(input_file_path) as img:
                    print(f"Processing {filename}...")

                    # a. Convert to greyscale
                    img_greyscale = img.convert('L')

                    # b. Resize the image
                    img_resized = img_greyscale.resize(size)

                    # c. Save to the new output path
                    img_resized.save(output_file_path)

                    print(f"Successfully converted and saved to {output_file_path}")

            except Exception as e:
                print(f"Could not process {filename}. Error: {e}")

# Run the conversion function with the new input/output folders
convert_to_greyscale_and_resize(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_SIZE)
print("Conversion process finished.")