import cv2
import config
import numpy as np
from wall_detection import detect_walls
from cad_export import export_walls_to_dxf


INPUT_IMAGE = "data/input/exemplo_rascunho.jpeg"
DEBUG_PATH = "data/debug/"
DXF_OUTPUT = "data/output/output.dxf"


def preprocess_image(image_path: str, blur_kernel: int, threshold: int):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    _, binary = cv2.threshold(
        blurred, threshold, 255, cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(cleaned, config.CANNY_LOW, config.CANNY_HIGH)

    return {
        "original": original,
        "gray": gray,
        "binary": binary,
        "edges": edges
    }


def draw_walls(image, walls):
    img = image.copy()

    for w in walls:
        if w.orientation == "horizontal":
            cv2.line(
                img,
                (int(w.start), int(w.coord)),
                (int(w.end), int(w.coord)),
                (0, 255, 0),
                3,
            )
        else:
            cv2.line(
                img,
                (int(w.coord), int(w.start)),
                (int(w.coord), int(w.end)),
                (0, 255, 0),
                3,
            )

    return img


def main():
    images = preprocess_image(
        INPUT_IMAGE, 
        blur_kernel=config.BLUR_KERNEL, 
        threshold=config.THRESHOLD
    )

    walls = detect_walls(images["edges"])

    debug = draw_walls(images["original"], walls)
    cv2.imwrite(f"{DEBUG_PATH}walls_detected.png", debug)

    export_walls_to_dxf(walls, DXF_OUTPUT)

    print(f"{len(walls)} paredes detectadas")


if __name__ == "__main__":
    main()
