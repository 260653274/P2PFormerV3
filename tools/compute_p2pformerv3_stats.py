#!/usr/bin/env python
"""Compute clean support-state statistics for P2PFormerV3.

The target construction mirrors Section 3.3 of the Micro Design and writes a
small JSON artifact that can be copied into a model config.  No image pixels
are loaded; only COCO polygon annotations are required.
"""

import argparse
import json
import math

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'annotation', help='COCO annotation JSON containing polygon masks')
    parser.add_argument('--output', required=True, help='Output statistics')
    parser.add_argument('--max-slots', type=int, default=40)
    parser.add_argument('--context-scale', type=float, default=1.1)
    parser.add_argument('--rho', type=float, default=6.0 / 32.0)
    parser.add_argument('--margin', type=float, default=2.0 / 32.0)
    parser.add_argument('--min-size', type=float, default=4.0 / 32.0)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def signed_area(points):
    shifted = np.roll(points, -1, axis=0)
    return 0.5 * np.sum(points[:, 0] * shifted[:, 1] -
                        shifted[:, 0] * points[:, 1])


def orientation(a, b, c, eps=1e-8):
    value = np.cross(b - a, c - a)
    if abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def on_segment(a, b, p, eps=1e-8):
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def segments_intersect(a, b, c, d):
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on_segment(a, b, c)) or
            (o2 == 0 and on_segment(a, b, d)) or
            (o3 == 0 and on_segment(c, d, a)) or
            (o4 == 0 and on_segment(c, d, b)))


def can_remove(points, index, winding):
    count = len(points)
    prev_index = (index - 1) % count
    next_index = (index + 1) % count
    a, b = points[prev_index], points[next_index]
    for edge_index in range(count):
        edge_next = (edge_index + 1) % count
        if edge_index in (prev_index, index, next_index):
            continue
        if edge_next in (prev_index, index, next_index):
            continue
        if segments_intersect(a, b, points[edge_index], points[edge_next]):
            return False
    reduced = np.delete(points, index, axis=0)
    area = signed_area(reduced)
    return abs(area) > 1e-8 and math.copysign(1.0, area) == winding


def simplify_closed_polygon(points, max_vertices):
    if len(points) <= max_vertices:
        return points
    winding = math.copysign(1.0, signed_area(points))
    while len(points) > max_vertices:
        previous = np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0)
        areas = np.abs(np.cross(points - previous, following - points)) * 0.5
        removed = False
        for index in np.argsort(areas):
            if can_remove(points, int(index), winding):
                points = np.delete(points, int(index), axis=0)
                removed = True
                break
        if not removed:
            raise RuntimeError('Could not simplify polygon without crossing')
    return points


def clean_polygon(flat_polygon):
    points = np.asarray(flat_polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < 1e-6:
        points = points[:-1]
    if len(points) == 0:
        return points
    previous = np.roll(points, 1, axis=0)
    points = points[np.linalg.norm(points - previous, axis=1) >= 0.1]
    return points


def support_states(points, context_scale, rho, margin, min_size):
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    size = np.maximum(maximum - minimum, 1e-6) * context_scale
    normalized = (points - (center - 0.5 * size)) / size

    current = normalized
    previous = np.roll(normalized, 1, axis=0)
    following = np.roll(normalized, -1, axis=0)

    def clip_incident(neighbor):
        vector = neighbor - current
        length = np.maximum(np.linalg.norm(vector, axis=1, keepdims=True),
                            1e-6)
        scale = np.minimum(1.0, rho / length)
        return current + vector * scale

    clipped_previous = clip_incident(previous)
    clipped_following = clip_incident(following)
    support_points = np.stack(
        (clipped_previous, current, clipped_following), axis=1)
    low = support_points.min(axis=1) - margin
    high = support_points.max(axis=1) + margin
    support_center = 0.5 * (low + high)
    support_size = np.maximum(high - low, min_size)
    states = np.concatenate(
        (support_center, np.log(support_size + 1e-6)), axis=1)
    return states


def main():
    args = parse_args()
    with open(args.annotation, encoding='utf-8') as file:
        coco = json.load(file)

    total = np.zeros(4, dtype=np.float64)
    total_square = np.zeros(4, dtype=np.float64)
    state_min = np.full(4, np.inf, dtype=np.float64)
    state_max = np.full(4, -np.inf, dtype=np.float64)
    state_count = 0
    component_count = 0
    simplified_count = 0

    for annotation in coco['annotations']:
        segmentation = annotation.get('segmentation', [])
        if not isinstance(segmentation, list):
            continue
        for flat_polygon in segmentation:
            points = clean_polygon(flat_polygon)
            if len(points) < 3:
                continue
            component_count += 1
            if len(points) > args.max_slots:
                points = simplify_closed_polygon(points, args.max_slots)
                simplified_count += 1
            states = support_states(points, args.context_scale, args.rho,
                                    args.margin, args.min_size)
            total += states.sum(axis=0)
            total_square += np.square(states).sum(axis=0)
            state_min = np.minimum(state_min, states.min(axis=0))
            state_max = np.maximum(state_max, states.max(axis=0))
            state_count += len(states)
            if args.limit and component_count >= args.limit:
                break
        if args.limit and component_count >= args.limit:
            break

    mean = total / max(state_count, 1)
    variance = total_square / max(state_count, 1) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    result = dict(
        annotation=args.annotation,
        component_count=component_count,
        simplified_component_count=simplified_count,
        active_support_count=state_count,
        state_order=['cx', 'cy', 'log_w', 'log_h'],
        mean=mean.tolist(),
        std=std.tolist(),
        minimum=state_min.tolist(),
        maximum=state_max.tolist(),
        settings=dict(
            max_slots=args.max_slots,
            context_scale=args.context_scale,
            rho=args.rho,
            margin=args.margin,
            min_size=args.min_size))
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
