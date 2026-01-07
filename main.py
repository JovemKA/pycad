import cv2
from src.preprocessing.image_cleaner import preprocess_image

INPUT_IMAGE = "data/input/exemplo_rascunho.jpeg"
DEBUG_PATH = "data/debug/"

def save_debug_images(images: dict):
  for name, img in images.items():
    path = f"{DEBUG_PATH}{name}.png"
    cv2.imwrite(path, img)
    print(f"Imagem salva: {path}")

def main():
  images = preprocess_image(INPUT_IMAGE)
  save_debug_images(images)

if __name__ == "__main__":
    main()