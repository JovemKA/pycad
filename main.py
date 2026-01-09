import cv2
from paddleocr import PaddleOCR
import config
import numpy as np
from wall_detection import detect_walls, separate_walls_and_dimensions
from cad_export import export_walls_to_dxf


INPUT_IMAGE = "data/input/rascunho_profissional.jpg"
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


def draw_lines(image, walls):
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


def draw_ocr_items(image, ocr_items):
    img = image.copy()

    for item in ocr_items:
        bbox = item["bbox"]
        text = item["text"]

        pts = np.array(bbox, dtype=np.int32)

        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(
            img,
            text,
            (pts[0][0], pts[0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
        )

    return img


def draw_walls_and_dimensions(image, real_walls, dimension_lines):
    img = image.copy()

    # 🟩 Paredes reais
    for w in real_walls:
        if w.orientation == "horizontal":
            cv2.line(
                img,
                (int(w.start), int(w.coord)),
                (int(w.end), int(w.coord)),
                (0, 255, 0),  # verde
                3,
            )
        else:
            cv2.line(
                img,
                (int(w.coord), int(w.start)),
                (int(w.coord), int(w.end)),
                (0, 255, 0),  # verde
                3,
            )

    # 🟦 Linhas de cota
    for w in dimension_lines:
        if w.orientation == "horizontal":
            cv2.line(
                img,
                (int(w.start), int(w.coord)),
                (int(w.end), int(w.coord)),
                (255, 0, 0),  # azul
                2,
            )
        else:
            cv2.line(
                img,
                (int(w.coord), int(w.start)),
                (int(w.coord), int(w.end)),
                (255, 0, 0),  # azul
                2,
            )

    return img


def main():
    # Preprocessamento
    images = preprocess_image(
        INPUT_IMAGE, 
        blur_kernel=config.BLUR_KERNEL, 
        threshold=config.THRESHOLD
    )

    # OCR
    ocr = PaddleOCR(
        lang='pt',
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

    result = ocr.predict(images["original"])

    ocr_items = []

    for r in result:
        texts = r.get("rec_texts", [])
        scores = r.get("rec_scores", [])
        polys = r.get("rec_polys", [])

        print(texts, scores)

        for text, score, poly in zip(texts, scores, polys):
            x_coords = [p[0] for p in poly]
            y_coords = [p[1] for p in poly]
            x_min, y_min = int(min(x_coords)), int(min(y_coords))
            x_max, y_max = int(max(x_coords)), int(max(y_coords))
            bbox = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

            ocr_items.append({
                "text": text,
                "score": score,
                "bbox": bbox,
            })

    # Detectar paredes
    walls = detect_walls(images["edges"])

    real_walls, dimension_lines = separate_walls_and_dimensions(
        walls, ocr_items
    )

    # Interpretar planta
        

    # CAD Export (FALTA AJUSTAR)
    export_walls_to_dxf(walls, DXF_OUTPUT)

    # Debug
    debug_lines = draw_lines(images["original"], walls)
    debug_ocr = draw_ocr_items(images["original"], ocr_items)
    debug_walls_and_dimensions = draw_walls_and_dimensions(
        images["original"], real_walls, dimension_lines
    )

    cv2.imwrite(f"{DEBUG_PATH}lines_detected.png", debug_lines)
    cv2.imwrite(f"{DEBUG_PATH}ocr_items.png", debug_ocr)
    cv2.imwrite(f"{DEBUG_PATH}walls_vs_dimensions.png", debug_walls_and_dimensions)

    print(f"{len(walls)} paredes detectadas")
    print("Paredes reais:", len(real_walls))
    print("Cotas:", len(dimension_lines))

if __name__ == "__main__":
    main()
