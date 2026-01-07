import cv2
import numpy as np
import math

def detect_walls(
    binary_image,
    canny_low=50,
    canny_high=150,
    hough_threshold=100,
    min_line_length=80,
    max_line_gap=10,
    angle_tolerance_deg=5
):
    """
    Detecta paredes (linhas horizontais e verticais).
    """

    # 1. Detectar bordas
    edges = cv2.Canny(
        binary_image,
        canny_low,
        canny_high
    )

    # 2. Detectar linhas
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    wall_lines = []

    if lines is None:
        return edges, wall_lines

    # 3. Filtrar linhas quase horizontais ou verticais
    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = abs(angle)

        # Horizontal (~0°) ou Vertical (~90°)
        if angle < angle_tolerance_deg or abs(angle - 90) < angle_tolerance_deg:
            wall_lines.append((x1, y1, x2, y2))

    return edges, wall_lines
