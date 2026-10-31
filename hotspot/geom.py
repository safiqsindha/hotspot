"""Polygon math in normalized scene coordinates (x right, y down, 0–1)."""

from __future__ import annotations


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def polygon_area(polygon: list[list[float]]) -> float:
    total = 0.0
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def centroid(polygon: list[list[float]]) -> tuple[float, float]:
    twice = 0.0
    cx = 0.0
    cy = 0.0
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = x1 * y2 - x2 * y1
        twice += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(twice) < 1e-12:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    cx /= 3.0 * twice
    cy /= 3.0 * twice
    if not point_in_polygon(cx, cy, polygon):
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return cx, cy


def box_covers_polygon(box: list[float], polygon: list[list[float]]) -> bool:
    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    for x, y in polygon:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    if any(point_in_polygon(x, y, polygon) for x, y in corners):
        return True
    poly_edges = _edges(polygon)
    rect_edges = _edges(corners)
    return any(_segments_intersect(edge, other) for edge in poly_edges for other in rect_edges)


def _edges(points) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pts = [(float(point[0]), float(point[1])) for point in points]
    return list(zip(pts, pts[1:] + pts[:1]))


def _segments_intersect(a, b) -> bool:
    (x1, y1), (x2, y2) = a
    (x3, y3), (x4, y4) = b

    def orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        return (by - ay) * (cx - bx) - (bx - ax) * (cy - by)

    o1 = orient(x1, y1, x2, y2, x3, y3)
    o2 = orient(x1, y1, x2, y2, x4, y4)
    o3 = orient(x3, y3, x4, y4, x1, y1)
    o4 = orient(x3, y3, x4, y4, x2, y2)
    if o1 == 0 or o2 == 0 or o3 == 0 or o4 == 0:
        return False
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
