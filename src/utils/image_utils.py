from PIL import Image

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def resize_image(image: Image.Image, size=(512, 512)) -> Image.Image:
    return image.resize(size)