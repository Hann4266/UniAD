import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, box, Polygon
from typing import Dict, List, Tuple, Optional, Union

# def get_patch_coord(patch_box,
#                     patch_angle = 0,
#                     fov_angle = 70):
#     patch_x, patch_y, _, y_max = patch_box
#     half_fov_rad = np.radians(fov_angle / 2.0)
    
#     # Triangle vertices in vehicle frame
#     apex = (patch_x, patch_y)  # Ego position
#     left_point = (patch_x + y_max, patch_y - y_max * np.tan(half_fov_rad))
#     right_point = (patch_x + y_max , patch_y + y_max* np.tan(half_fov_rad))
    
#     triangle = Polygon([apex, left_point, right_point])
    
#     # Rotate according to vehicle heading
#     triangle = affinity.rotate(triangle, patch_angle,
#                             origin=(patch_x, patch_y),
#                             use_radians=False)
    
#     return triangle

def get_patch_coord(patch_box: Tuple[float, float, float, float],
                        patch_angle: float = 0.0) -> Polygon:
        """
        Convert patch_box to shapely Polygon coordinates.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :return: Box Polygon for patch_box.
        """
        patch_x, patch_y, patch_h, patch_w = patch_box

        x_min = patch_x 
        y_min = patch_y - patch_w / 2.0
        x_max = patch_x + patch_h
        y_max = patch_y + patch_w / 2.0

        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

        return patch

def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0)

    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(
                coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx


def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)
    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w
    # trans_x = -patch_x + patch_w / 2.0
    # trans_y = -patch_y + patch_h / 2.0
    trans_x = -patch_x 
    trans_y = -patch_y + patch_w / 2.0
    map_mask = np.zeros((canvas_w,canvas_h), np.uint8)
    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(
                new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(
                new_line, xfact=scale_height, yfact=scale_width, origin=(0, 0))
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line:
                    map_mask, idx = mask_for_lines(
                        new_single_line, map_mask, thickness, idx, type, angle_class)
            elif new_line.geom_type == 'LineString':
                map_mask, idx = mask_for_lines(
                    new_line, map_mask, thickness, idx, type, angle_class)
            # Handle other geometry types (shouldn't happen, but just in case)
            # Handle regular LineString
            elif new_line.geom_type == 'LineString':
                map_mask, idx = mask_for_lines(
                    new_line, map_mask, thickness, idx, type, angle_class)
            
            # Handle GeometryCollection - extract only LineStrings
            elif new_line.geom_type == 'GeometryCollection':
                for geom in new_line.geoms:
                    if geom.geom_type == 'LineString':
                        map_mask, idx = mask_for_lines(
                            geom, map_mask, thickness, idx, type, angle_class)
                    elif geom.geom_type == 'MultiLineString':
                        for single_line in geom.geoms:
                            map_mask, idx = mask_for_lines(
                                single_line, map_mask, thickness, idx, type, angle_class)
                    # Skip Point, Polygon, etc. - not relevant for lines
            
            # Handle unexpected types
            else:
                print(f"Warning: Skipping unexpected geometry type: {new_line.geom_type}")
            # else:
            #     map_mask, idx = mask_for_lines(
            #         new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def preprocess_map(vectors, patch_size, canvas_size, num_classes, thickness, angle_class):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(
                LineString(vector['pts'][:vector['pts_num']]))
    # for line in vector_num_list[2]: 
    #     print(list(line.coords))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = overlap_filter(
        forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(
        backward_masks, filter_masks).sum(0).astype('int32')

    semantic_masks = instance_masks != 0

    return semantic_masks, instance_masks, forward_masks, backward_masks


def rasterize_map(vectors, patch_size, canvas_size, num_classes, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(
                (LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        masks.append(map_mask)

    return np.stack(masks), confidence_levels
