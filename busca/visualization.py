import cv2
import numpy as np


def plot_box(frame_image, target_id, target_bbox, style='solid', thickness=2, display_id=False, id_size=1.0, color=None):
    """style can either be 'solid', 'dashed' or 'dotted'"""
    if color is None:
        current_color = __choose_color(target_id)
    else:
        current_color = color
    target_bbox = np.array(target_bbox, dtype=np.int32)

    if style == 'dotted':  # Default arguments are too hard to see
        gap = 15
    else:
        gap = 20

    # Bounding-box
    drawrect(frame_image, tuple(target_bbox[:2]), tuple(target_bbox[2:]),
             current_color, thickness=thickness, style=style, gap=gap)

    # Object-center
    target_center = (target_bbox[:2] + target_bbox[2:]) / 2.0
    target_center = tuple(target_center.astype(np.int32))
    cv2.rectangle(frame_image,
                    target_center, target_center,
                    current_color, thickness=4)
    
    # Object-ID
    if display_id:
        cv2.putText(frame_image, str(target_id), tuple(target_bbox[:2]), cv2.FONT_HERSHEY_PLAIN, id_size, current_color, 2, cv2.LINE_AA)

def create_batch_image(image_list_mem, image_list_can, output_probs=None, show_hidden_tokens=True, max_batch_size=5, unnormalize=True, bgr_color=tuple([255, 255, 255])):
    image_list_mem = image_list_mem[:max_batch_size]
    image_list_can = image_list_can[:max_batch_size]

    if unnormalize:
        input_pixel_mean = np.array([0.406, 0.456, 0.485])  # BGR
        input_pixel_std = np.array([0.225, 0.224, 0.299])  # BGR

        image_list_mem = image_list_mem * input_pixel_std + input_pixel_mean
        image_list_mem = image_list_mem * 255
        image_list_mem = np.clip(image_list_mem, 0, 255).astype(np.uint8)

        image_list_can = image_list_can * input_pixel_std + input_pixel_mean
        image_list_can = image_list_can * 255
        image_list_can = np.clip(image_list_can, 0, 255).astype(np.uint8)

    batch_size = image_list_mem.shape[0]
    mem_len = image_list_mem.shape[1]

    if show_hidden_tokens and output_probs is not None:
        can_len = output_probs.shape[1]
    else:
        can_len = image_list_can.shape[1]

    if output_probs is not None:
        # We will write the output probabilities on the images
        output_probs = output_probs[:max_batch_size]
        new_image_list_can = []
        for b in range(batch_size):
            current_sample_cans = []
            for c in range(can_len):
                if c >= len(image_list_can[b]):
                    # We are on the hidden tokens (we assume they are the last ones in the list)
                    current_can = np.zeros_like(image_list_can[b, 0])
                else:
                    current_can = image_list_can[b, c]
                prob_label = create_value_image(label="Prob", value=str(round(output_probs[b, c], 3)), txt_color=tuple([0, 0, 0]), bgr_color=bgr_color, height=30,
                                                label_width=80, value_width=100, text_gap=5, label_scale=1.05, label_thickness=1,
                                                value_scale=1.05, value_thickness=1)
                current_can = vertical_stack(current_can, prob_label, separation=0, bgr_color=bgr_color)
                current_sample_cans.append(current_can)

            new_image_list_can.append(current_sample_cans)
            
        image_list_can = np.array(new_image_list_can)
    
    sample_img_list = []
    for b in range(batch_size):
        mem_img = image_list_mem[b, 0]  # Our base will be the first image in the list
        for m in range(1, mem_len):
            mem_img = horizontal_stack(mem_img, image_list_mem[b, m], separation=5, bgr_color=bgr_color)
        
        can_img = image_list_can[b, 0]  # Our base will be the first image in the list
        for c in range(1, can_len):
            can_img = horizontal_stack(can_img, image_list_can[b, c], separation=5, bgr_color=bgr_color)
        
        sample_img = horizontal_stack(mem_img, can_img, separation=10, bgr_color=tuple([255, 0, 0]))
        sample_img_list.append(sample_img)
    
    batch_img = sample_img_list[0]
    for s in range(1, len(sample_img_list)):
        batch_img = vertical_stack(batch_img, sample_img_list[s], separation=10, bgr_color=bgr_color)
    
    return batch_img

def __choose_color(i):
    current_color = _COLORS[int(i) % len(_COLORS)]
    current_color = tuple([int(c*255) for c in current_color])

    return current_color

def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i/dist
        x = int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y = int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)

    elif style == 'dashed':
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1
    
    elif style == 'solid':
        cv2.line(img, pt1, pt2, color, thickness)
    
    else:
        raise NotImplementedError

def drawpoly(img, pts, color, thickness=1, style='dotted', gap=20):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style, gap)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style, gap)


def create_value_image(label, value, txt_color=tuple([0, 0, 0]), bgr_color=tuple([255, 255, 255]), height=35,
                       label_width=230, value_width=190, text_gap=10, label_scale=1.05, label_thickness=1,
                       value_scale=1.05, value_thickness=1):
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, label_scale, label_thickness)

    label_blue_channel = np.full((height, label_width), bgr_color[0]).astype(np.uint8)
    label_green_channel = np.full((height, label_width), bgr_color[1]).astype(np.uint8)
    label_red_channel = np.full((height, label_width), bgr_color[2]).astype(np.uint8)
    label_blank = np.stack((label_blue_channel, label_green_channel, label_red_channel), axis=2)

    cv2.putText(label_blank, label, tuple([label_width - label_size[0][0], int((height / 2) + (label_size[0][1] / 2))]),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, label_scale, txt_color, label_thickness, cv2.LINE_AA)

    value_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_COMPLEX_SMALL, value_scale, value_thickness)
    value_blue_channel = np.full((height, value_width), bgr_color[0]).astype(np.uint8)
    value_green_channel = np.full((height, value_width), bgr_color[1]).astype(np.uint8)
    value_red_channel = np.full((height, value_width), bgr_color[2]).astype(np.uint8)
    value_blank = np.stack((value_blue_channel, value_green_channel, value_red_channel), axis=2)

    cv2.putText(value_blank, value, tuple([0, int((height / 2) + (value_size[0][1] / 2))]), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                value_scale, txt_color, value_thickness, cv2.LINE_AA)

    image = horizontal_stack(label_blank, value_blank, separation=text_gap, bgr_color=bgr_color)
    return image


def horizontal_stack(left_image, right_image, separation=5, bgr_color=tuple([255, 255, 255])):
    lh, lw, _ = left_image.shape
    rh, rw, _ = right_image.shape

    if lh > rh:
        difference = lh - rh
        if difference % 2 == 0:
            top_padding = difference // 2
            bottom_padding = top_padding
        else:
            top_padding = difference // 2
            bottom_padding = top_padding + 1

        right_image = cv2.copyMakeBorder(right_image, top_padding, bottom_padding, separation, 0, cv2.BORDER_CONSTANT, value=bgr_color)

    elif lh < rh:
        difference = rh - lh
        if difference % 2 == 0:
            top_padding = difference // 2
            bottom_padding = top_padding
        else:
            top_padding = difference // 2
            bottom_padding = top_padding + 1

        left_image = cv2.copyMakeBorder(left_image, top_padding, bottom_padding, 0, separation, cv2.BORDER_CONSTANT, value=bgr_color)

    else:
        left_image = cv2.copyMakeBorder(left_image, 0, 0, 0, separation, cv2.BORDER_CONSTANT, value=bgr_color)

    image = np.hstack((left_image, right_image))

    return image


def vertical_stack(top_image, bottom_image, separation=5, bgr_color=tuple([255, 255, 255])):
    th, tw, _ = top_image.shape
    bh, bw, _ = bottom_image.shape

    if tw > bw:
        difference = tw - bw
        if difference % 2 == 0:
            left_padding = difference // 2
            right_padding = left_padding
        else:
            left_padding = difference // 2
            right_padding = left_padding + 1

        bottom_image = cv2.copyMakeBorder(bottom_image, separation, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=bgr_color)

    elif tw < bw:
        difference = bw - tw
        if difference % 2 == 0:
            left_padding = difference // 2
            right_padding = left_padding
        else:
            left_padding = difference // 2
            right_padding = left_padding + 1

        top_image = cv2.copyMakeBorder(top_image, 0, separation, left_padding, right_padding, cv2.BORDER_CONSTANT, value=bgr_color)

    else:
        top_image = cv2.copyMakeBorder(top_image, 0, separation, 0, 0, cv2.BORDER_CONSTANT, value=bgr_color)

    image = np.vstack((top_image, bottom_image))

    return image

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)