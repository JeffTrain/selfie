"""LEGO 小人仔头部渲染模块

根据人脸 68 特征点，在图像上绘制个性化 LEGO 小人仔风格的头部：
- 基于真实肤色的头部
- 动态跟踪的眼睛开合
- 跟随真实表情的嘴型（闭合/张开/微笑）
- 跟踪真实眉毛角度的眉毛
- 从真实头发提取的发色头发
- 顶部圆柱形凸点 (stud)
"""

import cv2
import math
import numpy as np


# LEGO 经典颜色 (BGR 格式)
LEGO_YELLOW = (0, 215, 255)        # 头部黄色
LEGO_YELLOW_DARK = (0, 180, 220)   # 头部暗部/阴影
LEGO_BLACK = (0, 0, 0)             # 眼睛和嘴巴
LEGO_OUTLINE = (0, 140, 180)       # 轮廓线（深黄色）
LEGO_STUD_TOP = (30, 230, 255)     # 凸点顶部（亮黄色）


def compute_eye_openness(landmarks):
    """
    计算左右眼的睁开程度 (0=全闭, 1=全开)。

    通过上下眼睑特征点的纵向距离与眼睛水平宽度的比值来衡量。
    """
    # 左眼: 37,38(上), 40,41(下), 36,39(左右)
    left_w = abs(landmarks[39][0] - landmarks[36][0])
    left_h = ((landmarks[40][1] + landmarks[41][1]) / 2 -
              (landmarks[37][1] + landmarks[38][1]) / 2)
    left_ratio = max(0, left_h) / max(left_w, 1)

    # 右眼: 43,44(上), 46,47(下), 42,45(左右)
    right_w = abs(landmarks[45][0] - landmarks[42][0])
    right_h = ((landmarks[46][1] + landmarks[47][1]) / 2 -
               (landmarks[43][1] + landmarks[44][1]) / 2)
    right_ratio = max(0, right_h) / max(right_w, 1)

    # 归一化: 正常睁眼比值约 0.25~0.5，映射到 0~1
    left_open = min(1.0, max(0.0, left_ratio / 0.5))
    right_open = min(1.0, max(0.0, right_ratio / 0.5))

    return left_open, right_open


def compute_mouth_openness(landmarks):
    """
    计算嘴巴张开程度 (0=闭合, 1=大张)。

    通过内唇上下距离与嘴巴宽度的比值来衡量。
    """
    # 内唇上 (62) 和 内唇下 (66)
    inner_h = abs(landmarks[66][1] - landmarks[62][1])
    # 嘴巴宽度
    mouth_w = abs(landmarks[54][0] - landmarks[48][0])

    ratio = inner_h / max(mouth_w, 1)
    # 归一化: 闭嘴约 0.05, 大张约 0.5+
    openness = min(1.0, max(0.0, ratio / 0.45))
    return openness


def compute_brow_angles(landmarks):
    """
    计算左右眉毛的倾斜角度（度数）。

    返回:
        (left_angle, right_angle): 角度值，0 为水平
    """
    # 左眉: 17->21
    left_dx = landmarks[21][0] - landmarks[17][0]
    left_dy = landmarks[21][1] - landmarks[17][1]
    left_angle = math.degrees(math.atan2(left_dy, max(abs(left_dx), 1)))

    # 右眉: 22->26
    right_dx = landmarks[26][0] - landmarks[22][0]
    right_dy = landmarks[26][1] - landmarks[22][1]
    right_angle = math.degrees(math.atan2(right_dy, max(abs(right_dx), 1)))

    return left_angle, right_angle


def sample_skin_color(frame, landmarks):
    """
    从人脸区域采样肤色。

    取鼻子到脸颊区间的像素中位数作为肤色。
    """
    img_h, img_w = frame.shape[:2]

    # 采样点：鼻梁 (27)、左脸颊 (1,2)、右脸颊 (14,15)
    sample_points = [landmarks[27], landmarks[1], landmarks[2],
                     landmarks[14], landmarks[15], landmarks[29],
                     landmarks[30]]

    colors = []
    radius = 3
    for pt in sample_points:
        x, y = int(pt[0]), int(pt[1])
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(img_w, x + radius + 1)
        y2 = min(img_h, y + radius + 1)
        if x2 > x1 and y2 > y1:
            patch = frame[y1:y2, x1:x2]
            colors.append(np.mean(patch, axis=(0, 1)))

    if not colors:
        return LEGO_YELLOW  # 兜底使用默认黄色

    median_color = np.median(colors, axis=0).astype(int)
    return tuple(int(c) for c in median_color)


def sample_hair_color(frame, landmarks):
    """
    从额头上方区域采样头发颜色。
    """
    img_h, img_w = frame.shape[:2]

    # 额头上方：取眉毛最高点再往上
    brow_y = min(landmarks[17][1], landmarks[26][1])
    center_x = (landmarks[0][0] + landmarks[16][0]) // 2
    face_width = abs(landmarks[16][0] - landmarks[0][0])

    # 采样区域：眉毛上方 20~50 像素
    hair_top = max(0, brow_y - 50)
    hair_bot = max(0, brow_y - 10)
    hair_left = max(0, center_x - face_width // 3)
    hair_right = min(img_w, center_x + face_width // 3)

    if hair_bot <= hair_top or hair_right <= hair_left:
        return (30, 30, 50)  # 默认深色

    patch = frame[hair_top:hair_bot, hair_left:hair_right]
    if patch.size == 0:
        return (30, 30, 50)

    median_color = np.median(patch.reshape(-1, 3), axis=0).astype(int)
    return tuple(int(c) for c in median_color)


def extract_face_features(frame, landmarks):
    """
    综合提取面部个性化特征。

    返回:
        dict: 包含所有个性化特征
    """
    left_open, right_open = compute_eye_openness(landmarks)
    mouth_open = compute_mouth_openness(landmarks)
    left_brow, right_brow = compute_brow_angles(landmarks)
    skin = sample_skin_color(frame, landmarks)
    hair = sample_hair_color(frame, landmarks)

    return {
        'eye_openness_left': left_open,
        'eye_openness_right': right_open,
        'mouth_openness': mouth_open,
        'brow_angle_left': left_brow,
        'brow_angle_right': right_brow,
        'skin_color': skin,
        'hair_color': hair,
    }


def compute_lego_geometry(landmarks):
    """
    从 68 个人脸特征点计算 LEGO 头部的几何参数。

    参数:
        landmarks: numpy array (68, 2)，人脸 68 个特征点坐标

    返回:
        dict: 包含 LEGO 头部绘制所需的几何参数
    """
    # 人脸宽度：从下巴轮廓左端 (0) 到右端 (16)
    face_left = landmarks[0][0]
    face_right = landmarks[16][0]
    face_width = abs(face_right - face_left)

    # 人脸中心
    face_center_x = (face_left + face_right) // 2

    # 眉毛到下巴的纵向范围
    brow_y = min(landmarks[17][1], landmarks[26][1])
    chin_y = landmarks[8][1]
    face_height = abs(chin_y - brow_y)

    # LEGO 头部尺寸：比人脸略大，圆柱形比例
    head_width = int(face_width * 1.3)
    head_height = int(head_width * 1.25)  # LEGO 头部高宽比约 1.25

    # 头部中心：基于眉毛和下巴的中间位置
    face_center_y = (brow_y + chin_y) // 2

    # 左右眼中心
    left_eye_x = (landmarks[36][0] + landmarks[39][0]) // 2
    left_eye_y = (landmarks[37][1] + landmarks[41][1]) // 2
    right_eye_x = (landmarks[42][0] + landmarks[45][0]) // 2
    right_eye_y = (landmarks[43][1] + landmarks[47][1]) // 2

    # 将眼睛映射到 LEGO 头部的相对位置
    # 眼睛在头部上半部分，水平间距加大
    eye_spacing = int(head_width * 0.28)
    eye_y_offset = int(head_height * -0.05)  # 略在中心偏上

    lego_left_eye = (face_center_x - eye_spacing, face_center_y + eye_y_offset)
    lego_right_eye = (face_center_x + eye_spacing, face_center_y + eye_y_offset)

    # 嘴巴
    mouth_left = landmarks[48]
    mouth_right = landmarks[54]
    mouth_center_x = (mouth_left[0] + mouth_right[0]) // 2
    mouth_center_y = int(face_center_y + head_height * 0.2)
    mouth_width = int(head_width * 0.4)

    # 凸点 (stud) - 在头顶
    stud_center = (face_center_x, face_center_y - head_height // 2 - int(head_width * 0.08))
    stud_radius = int(head_width * 0.15)

    # 眉毛位置
    brow_left_pos = (face_center_x - eye_spacing, face_center_y + eye_y_offset - int(head_width * 0.12))
    brow_right_pos = (face_center_x + eye_spacing, face_center_y + eye_y_offset - int(head_width * 0.12))
    brow_width = int(head_width * 0.18)

    return {
        'center_x': face_center_x,
        'center_y': face_center_y,
        'head_width': head_width,
        'head_height': head_height,
        'left_eye': lego_left_eye,
        'right_eye': lego_right_eye,
        'mouth_center': (mouth_center_x, mouth_center_y),
        'mouth_width': mouth_width,
        'stud_center': stud_center,
        'stud_radius': stud_radius,
        'brow_left': brow_left_pos,
        'brow_right': brow_right_pos,
        'brow_width': brow_width,
    }


def _clip_region(x, y, w, h, img_h, img_w):
    """将区域裁剪到图像边界内"""
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    return x1, y1, x2, y2


def draw_lego_head(frame, landmarks):
    """
    在图像上绘制个性化 LEGO 小人仔头部，覆盖真实人脸。

    个性化特征包括：
    - 真实肤色作为头部底色
    - 眼睛开合跟随真实眼睛
    - 嘴型跟随真实嘴巴（闭合/张开）
    - 眉毛角度跟随真实眉毛
    - 头发颜色来自真实头发

    参数:
        frame: numpy array (H, W, 3)，BGR 格式图像
        landmarks: numpy array (68, 2)，人脸 68 个特征点坐标

    返回:
        frame: 绘制了 LEGO 头部的图像
    """
    geo = compute_lego_geometry(landmarks)
    features = extract_face_features(frame, landmarks)
    img_h, img_w = frame.shape[:2]

    cx = geo['center_x']
    cy = geo['center_y']
    hw = geo['head_width']
    hh = geo['head_height']

    # 使用真实肤色（与 LEGO 黄色混合，保持 LEGO 风格）
    skin = features['skin_color']
    head_color = _blend_color(skin, LEGO_YELLOW, 0.35)
    head_outline = _darken_color(head_color, 0.6)

    # --- 1. 绘制头部主体 ---
    half_w = hw // 2
    half_h = hh // 2

    top_left = (cx - half_w, cy - half_h)
    bottom_right = (cx + half_w, cy + half_h)
    corner_radius = int(hw * 0.18)

    _draw_rounded_rect(frame, top_left, bottom_right, corner_radius,
                       head_color, thickness=-1)

    outline_thickness = max(2, int(hw * 0.025))
    _draw_rounded_rect(frame, top_left, bottom_right, corner_radius,
                       head_outline, thickness=outline_thickness)

    # --- 2. 绘制头发 ---
    hair_color = features['hair_color']
    hair_height = int(hh * 0.22)
    hair_top = cy - half_h - 2
    hair_pts = np.array([
        [cx - half_w - 3, hair_top + hair_height],
        [cx - half_w - 5, hair_top - 2],
        [cx - half_w // 2, hair_top - hair_height // 2],
        [cx, hair_top - hair_height // 3],
        [cx + half_w // 2, hair_top - hair_height // 2],
        [cx + half_w + 5, hair_top - 2],
        [cx + half_w + 3, hair_top + hair_height],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [hair_pts], hair_color, cv2.LINE_AA)
    cv2.polylines(frame, [hair_pts], False, _darken_color(hair_color, 0.5),
                  max(1, int(hw * 0.02)), cv2.LINE_AA)

    # --- 3. 绘制眼睛（动态开合） ---
    eye_max_radius = max(3, int(hw * 0.065))
    lx, ly = geo['left_eye']
    rx, ry = geo['right_eye']

    left_open = features['eye_openness_left']
    right_open = features['eye_openness_right']

    _draw_lego_eye(frame, lx, ly, eye_max_radius, left_open, hw)
    _draw_lego_eye(frame, rx, ry, eye_max_radius, right_open, hw)

    # --- 4. 绘制眉毛 ---
    brow_lx, brow_ly = geo['brow_left']
    brow_rx, brow_ry = geo['brow_right']
    brow_w = geo['brow_width']
    brow_thickness = max(2, int(hw * 0.035))

    left_brow_angle = features['brow_angle_left']
    right_brow_angle = features['brow_angle_right']

    _draw_lego_brow(frame, brow_lx, brow_ly, brow_w, left_brow_angle,
                    brow_thickness, head_outline)
    _draw_lego_brow(frame, brow_rx, brow_ry, brow_w, right_brow_angle,
                    brow_thickness, head_outline)

    # --- 5. 绘制嘴巴（动态开合） ---
    mx, my = geo['mouth_center']
    mw = geo['mouth_width']
    mouth_open = features['mouth_openness']

    _draw_lego_mouth(frame, mx, my, mw, mouth_open, hw)

    # --- 6. 绘制顶部凸点 (stud) ---
    sx, sy = geo['stud_center']
    sr = geo['stud_radius']
    stud_h = int(sr * 0.6)

    stud_color = _blend_color(head_color, LEGO_STUD_TOP, 0.5)
    cv2.ellipse(frame, (sx, sy), (sr, stud_h), 0, 0, 360,
                stud_color, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (sx, sy), (sr, stud_h), 0, 0, 360,
                head_outline, max(1, int(hw * 0.02)), cv2.LINE_AA)

    return frame


def _blend_color(color1, color2, alpha):
    """混合两个 BGR 颜色，alpha 为 color2 的权重"""
    return tuple(int(c1 * (1 - alpha) + c2 * alpha) for c1, c2 in zip(color1, color2))


def _darken_color(color, factor):
    """将颜色调暗"""
    return tuple(int(c * factor) for c in color)


def _draw_lego_eye(frame, cx, cy, max_radius, openness, head_width):
    """
    绘制单个 LEGO 眼睛，根据开合度改变形状。

    全开: 大黑圆点 + 白色高光
    半开: 椭圆，纵向压缩
    全闭: 水平线
    """
    if openness < 0.15:
        # 全闭 - 水平线
        line_w = max_radius + 2
        cv2.line(frame, (cx - line_w, cy), (cx + line_w, cy),
                 LEGO_BLACK, max(2, int(head_width * 0.025)), cv2.LINE_AA)
    else:
        # 纵向半径随开合度变化
        v_radius = max(2, int(max_radius * min(1.0, openness)))
        h_radius = max_radius

        # 黑色眼珠（椭圆）
        cv2.ellipse(frame, (cx, cy), (h_radius, v_radius), 0, 0, 360,
                    LEGO_BLACK, -1, cv2.LINE_AA)

        # 高光点（让眼睛有灵气）
        if openness > 0.4:
            highlight_r = max(1, int(max_radius * 0.3))
            hx = cx - int(max_radius * 0.25)
            hy = cy - int(v_radius * 0.3)
            cv2.circle(frame, (hx, hy), highlight_r, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_lego_brow(frame, cx, cy, width, angle, thickness, color):
    """
    绘制单条 LEGO 眉毛，根据角度倾斜。
    """
    half_w = width // 2
    rad = math.radians(angle)
    dx = int(half_w * math.cos(rad))
    dy = int(half_w * math.sin(rad))

    pt1 = (cx - dx, cy - dy)
    pt2 = (cx + dx, cy + dy)
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def _draw_lego_mouth(frame, cx, cy, width, openness, head_width):
    """
    绘制 LEGO 嘴巴，根据张开程度动态变化。

    闭合: 经典弧形微笑线
    微张: 小椭圆开口
    大张: 大椭圆开口
    """
    line_thickness = max(2, int(head_width * 0.03))
    half_w = width // 2

    if openness < 0.2:
        # 闭嘴 - 经典微笑弧线
        smile_height = int(width * 0.3)
        cv2.ellipse(frame,
                    (cx, cy - smile_height // 2),
                    (half_w, smile_height),
                    0, 10, 170,
                    LEGO_BLACK,
                    line_thickness,
                    cv2.LINE_AA)
    else:
        # 张嘴 - 黑色椭圆（嘴巴内部）
        mouth_h = max(3, int(width * 0.15 + width * 0.5 * openness))
        mouth_w = int(half_w * (0.6 + 0.4 * openness))

        # 黑色嘴巴内部
        cv2.ellipse(frame, (cx, cy), (mouth_w, mouth_h), 0, 0, 360,
                    LEGO_BLACK, -1, cv2.LINE_AA)

        # 如果张很大，加红色舌头暗示
        if openness > 0.5:
            tongue_w = int(mouth_w * 0.5)
            tongue_h = int(mouth_h * 0.4)
            tongue_y = cy + int(mouth_h * 0.3)
            cv2.ellipse(frame, (cx, tongue_y), (tongue_w, tongue_h), 0, 0, 360,
                        (80, 80, 180), -1, cv2.LINE_AA)


def _draw_rounded_rect(img, top_left, bottom_right, radius, color, thickness=-1):
    """
    绘制圆角矩形。

    参数:
        img: 目标图像
        top_left: (x, y) 左上角
        bottom_right: (x, y) 右下角
        radius: 圆角半径
        color: BGR 颜色
        thickness: 线宽，-1 为填充
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # 限制圆角半径
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    radius = max(radius, 0)

    if thickness == -1:
        # 填充模式：用多个矩形和圆形拼接
        # 中间矩形
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        # 左右矩形
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        # 四个圆角
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    else:
        # 轮廓模式
        # 上下直线
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        # 左右直线
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        # 四个圆角弧
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius),
                    180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius),
                    270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),
                    90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),
                    0, 0, 90, color, thickness, cv2.LINE_AA)
