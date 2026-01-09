import cv2
from dataclasses import dataclass
from collections import defaultdict
import config


@dataclass
class Wall:
    orientation: str  # "horizontal" | "vertical"
    coord: float  # y para horizontal, x para vertical
    start: int
    end: int


@dataclass
class DimensionLine:
    orientation: str
    coord: float
    start: int
    end: int
    value: str | None = None
    score: float | None = None
    target_wall: Wall | None = None


def detect_walls(edges):
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=3.14159 / 180,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.MIN_LINE_LENGTH,
        maxLineGap=config.MAX_LINE_GAP,
    )

    if lines is None:
        return []

    raw_lines = [l[0] for l in lines]
    normalized = normalize_lines(raw_lines)
    grouped = group_colinear_lines(normalized)
    merged = merge_groups(grouped)

    return merged


def separate_walls_and_dimensions(walls, ocr_items):
    real_walls = []
    dimension_lines = []

    for w in walls:
        matched_dimension = None

        for item in ocr_items:
            if not is_numeric_dimension(item["text"]):
                continue

            if not is_text_parallel_to_wall(item["bbox"], w):
                continue

            dist = distance_text_to_wall(item["bbox"], w)
            if dist > config.MAX_DIMENSION_OFFSET:
                continue

            matched_dimension = DimensionLine(
                orientation=w.orientation,
                coord=w.coord,
                start=w.start,
                end=w.end,
                value=item["text"],
                score=item["score"],
                target_wall=w,
            )
            break

        if matched_dimension:
            dimension_lines.append(matched_dimension)
        else:
            real_walls.append(w)

    return real_walls, dimension_lines


def normalize_lines(lines):
    """Transforma segmentos em linhas horizontais ou verticais"""
    norm = []

    for x1, y1, x2, y2 in lines:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > dy:
            y = (y1 + y2) / 2
            norm.append(
                {
                    "orientation": "horizontal",
                    "coord": y,
                    "start": min(x1, x2),
                    "end": max(x1, x2),
                }
            )
        elif dy > dx:
            x = (x1 + x2) / 2
            norm.append(
                {
                    "orientation": "vertical",
                    "coord": x,
                    "start": min(y1, y2),
                    "end": max(y1, y2),
                }
            )

    return norm


def group_colinear_lines(lines):
    """Agrupa linhas por orientação e proximidade"""
    groups = defaultdict(list)

    for line in lines:
        key = (line["orientation"], quantize(line["coord"]))
        groups[key].append(line)

    return groups


def quantize(value):
    """Agrupa coordenadas próximas"""
    tol = config.COLINEAR_TOLERANCE
    return int(value / tol) * tol


def merge_groups(groups):
    walls = []

    for (orientation, coord), lines in groups.items():
        lines = sorted(lines, key=lambda l: l["start"])

        current_start = lines[0]["start"]
        current_end = lines[0]["end"]

        for line in lines[1:]:
            if line["start"] <= current_end + config.MERGE_GAP:
                current_end = max(current_end, line["end"])
            else:
                if current_end - current_start >= config.MIN_WALL_AXIS_LENGTH:
                    walls.append(
                        Wall(orientation, coord, current_start, current_end)
                      )
                current_start = line["start"]
                current_end = line["end"]

        if current_end - current_start >= config.MIN_WALL_AXIS_LENGTH:
            walls.append(
                Wall(orientation, coord, current_start, current_end)
              )

    return walls


def is_numeric_dimension(text: str) -> bool:
    try:
        float(text.replace(",", "."))
        return True
    except ValueError:
        return False


def is_text_parallel_to_wall(bbox, wall: Wall) -> bool:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    if wall.orientation == "horizontal":
        return width > height
    else:
        return height > width


def distance_text_to_wall(bbox, wall: Wall) -> float:
    cx = sum(p[0] for p in bbox) / 4
    cy = sum(p[1] for p in bbox) / 4

    if wall.orientation == "horizontal":
        return abs(cy - wall.coord)
    else:
        return abs(cx - wall.coord)
