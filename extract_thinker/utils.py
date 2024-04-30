import base64
from PIL import Image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_type(image_path):
    try:
        img = Image.open(image_path)
        return img.format.lower()
    except IOError as e:
        return f"An error occurred: {str(e)}"
