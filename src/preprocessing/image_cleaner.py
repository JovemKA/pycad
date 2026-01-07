import cv2
import numpy as np

def preprocess_image(
    image_path: str,
    blur_kernel: int,
    threshold: int
):
    """
    Lê um rascunho e retorna uma imagem binarizada e limpa.
    """

    # 1. Ler imagem
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # 2. Converter para tons de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Redução de ruído (blur)
    blurred = cv2.GaussianBlur(
        gray,
        (blur_kernel, blur_kernel),
        0
    )

    # 4. Binarização (preto e branco)
    _, binary = cv2.threshold(
        blurred,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    # 5. Operação morfológica (remove sujeira pequena)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return {
        "original": image,
        "gray": gray,
        "binary": binary,
        "cleaned": cleaned
    }
