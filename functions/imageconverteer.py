from PIL import Image
import os

class ImageProcessor:
    def __init__(self, input_folder):
        self.input_folder = input_folder

    def convert_tif_to_8bit(self):
        files = os.listdir(self.input_folder)
        for file in files:
            if file.lower().endswith(".tif") or file.lower().endswith(".tiff"):
                input_path = os.path.join(self.input_folder, file)
                with Image.open(input_path) as img:
                    img_8bit = img.convert("L")
                    img_8bit_path = os.path.join(self.input_folder, f"{os.path.splitext(file)[0]}.tif")
                    img_8bit.save(img_8bit_path)

    def invert_colors(self):
        files = os.listdir(self.input_folder)
        for file in files:
            if file.lower().endswith(".tif") or file.lower().endswith(".tiff"):
                input_path = os.path.join(self.input_folder, file)
                with Image.open(input_path) as img:
                    inverted_img = Image.eval(img, lambda x: 255 - x)
                    inverted_img_path = os.path.join(self.input_folder, f"{os.path.splitext(file)[0]}.tif")
                    inverted_img.save(inverted_img_path)


# input_folder_path = "stitched_outputs"

# processor = ImageProcessor(input_folder_path)
# processor.convert_tif_to_8bit()
# processor.invert_colors()
