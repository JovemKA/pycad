import cv2
from src.preprocessing.image_cleaner import preprocess_image
from src.walls import detect_walls

INPUT_IMAGE = "data/input/exemplo_rascunho.jpeg"
DEBUG_PATH = "data/debug/"

def draw_lines(image, lines):
  img = image.copy()
  for x1, y1, x2, y2 in lines:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
  return img

def save_debug_images(images: dict):
  for name, img in images.items():
    path = f"{DEBUG_PATH}{name}.png"
    cv2.imwrite(path, img)
    print(f"Imagem salva: {path}")

def main():
  images = preprocess_image(
    INPUT_IMAGE,
    blur_kernel=5,
    threshold=100
  )
  
  edges, walls = detect_walls(
    images["cleaned"],
    canny_low=30,
    canny_high=150,
    hough_threshold=120,
    min_line_length=120,
    max_line_gap=30
  )

  cv2.imwrite(f"{DEBUG_PATH}edges.png", edges)

  walls_img = draw_lines(images["original"], walls)
  cv2.imwrite(f"{DEBUG_PATH}walls_detected.png", walls_img)

  print(f"Linhas detectadas: {len(walls)}")

if __name__ == "__main__":
    main()