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
    collapsed = collapse_parallel_walls(merged)

    return collapsed


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


def collapse_parallel_walls(walls):
    result = []
    used = set()

    for i, w1 in enumerate(walls):
        if i in used:
            continue

        group = [w1]
        used.add(i)

        for j, w2 in enumerate(walls[i + 1 :], start=i + 1):
            if j in used:
                continue

            if w1.orientation != w2.orientation:
                continue

            if abs(w1.coord - w2.coord) > config.MAX_WALL_THICKNESS:
                continue

            overlap = min(w1.end, w2.end) - max(w1.start, w2.start)
            min_len = min(w1.end - w1.start, w2.end - w2.start)

            if overlap > 0 and overlap / min_len >= config.MIN_OVERLAP_RATIO:
                group.append(w2)
                used.add(j)

        if len(group) == 1:
            result.append(w1)
        else:
            coord = sum(w.coord for w in group) / len(group)
            start = min(w.start for w in group)
            end = max(w.end for w in group)
            result.append(
                Wall(w1.orientation, coord, start, end)
              )

    return result
