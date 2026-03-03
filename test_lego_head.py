import cv2
import numpy as np
import pytest

from lego_head import (
    draw_lego_head, compute_lego_geometry,
    extract_face_features, compute_eye_openness, compute_mouth_openness,
    compute_brow_angles, sample_skin_color, sample_hair_color,
)


class TestComputeLegoGeometry:
    """测试从人脸特征点计算 LEGO 头部几何参数"""

    def _make_landmarks(self, center_x=200, center_y=200, face_width=100):
        """构造模拟的 68 个人脸特征点"""
        landmarks = np.zeros((68, 2), dtype=int)
        half_w = face_width // 2

        # 关键特征点
        # 0-16: 下巴轮廓
        landmarks[0] = (center_x - half_w, center_y)
        landmarks[16] = (center_x + half_w, center_y)
        landmarks[8] = (center_x, center_y + half_w)  # 下巴底部

        # 17-21: 左眉毛
        landmarks[17] = (center_x - half_w + 10, center_y - 30)
        landmarks[21] = (center_x - 10, center_y - 30)

        # 22-26: 右眉毛
        landmarks[22] = (center_x + 10, center_y - 30)
        landmarks[26] = (center_x + half_w - 10, center_y - 30)

        # 27: 鼻梁顶部（眉间）
        landmarks[27] = (center_x, center_y - 20)

        # 36-41: 左眼
        landmarks[36] = (center_x - 30, center_y - 15)
        landmarks[39] = (center_x - 10, center_y - 15)
        landmarks[37] = (center_x - 25, center_y - 20)
        landmarks[38] = (center_x - 15, center_y - 20)
        landmarks[40] = (center_x - 15, center_y - 10)
        landmarks[41] = (center_x - 25, center_y - 10)

        # 42-47: 右眼
        landmarks[42] = (center_x + 10, center_y - 15)
        landmarks[45] = (center_x + 30, center_y - 15)
        landmarks[43] = (center_x + 15, center_y - 20)
        landmarks[44] = (center_x + 25, center_y - 20)
        landmarks[46] = (center_x + 25, center_y - 10)
        landmarks[47] = (center_x + 15, center_y - 10)

        # 48-67: 嘴巴
        landmarks[48] = (center_x - 20, center_y + 15)  # 左嘴角
        landmarks[54] = (center_x + 20, center_y + 15)  # 右嘴角
        landmarks[51] = (center_x, center_y + 12)  # 上唇中心
        landmarks[57] = (center_x, center_y + 20)  # 下唇中心
        landmarks[62] = (center_x, center_y + 16)
        landmarks[66] = (center_x, center_y + 18)

        # 30, 33: 鼻尖
        landmarks[30] = (center_x, center_y)
        landmarks[33] = (center_x, center_y + 5)

        return landmarks

    def test_returns_dict_with_required_keys(self):
        """应返回包含所有必要几何参数的字典"""
        landmarks = self._make_landmarks()
        geo = compute_lego_geometry(landmarks)

        required_keys = ['center_x', 'center_y', 'head_width', 'head_height',
                         'left_eye', 'right_eye', 'mouth_center', 'mouth_width',
                         'stud_center', 'stud_radius']
        for key in required_keys:
            assert key in geo, f"缺少键: {key}"

    def test_head_dimensions_proportional_to_face(self):
        """LEGO 头的尺寸应与人脸宽度成比例"""
        landmarks = self._make_landmarks(face_width=100)
        geo = compute_lego_geometry(landmarks)

        # 头宽应大于人脸宽度（覆盖整个脸）
        assert geo['head_width'] >= 100
        # 高度应大于宽度（LEGO 头是竖直的圆柱体）
        assert geo['head_height'] > geo['head_width'] * 0.9

    def test_eyes_inside_head_bounds(self):
        """两只眼睛应在头部范围内"""
        landmarks = self._make_landmarks(center_x=200, center_y=200)
        geo = compute_lego_geometry(landmarks)

        half_w = geo['head_width'] // 2
        half_h = geo['head_height'] // 2

        for eye in [geo['left_eye'], geo['right_eye']]:
            assert geo['center_x'] - half_w <= eye[0] <= geo['center_x'] + half_w
            assert geo['center_y'] - half_h <= eye[1] <= geo['center_y'] + half_h

    def test_stud_on_top_of_head(self):
        """凸点应在头部顶端"""
        landmarks = self._make_landmarks(center_x=200, center_y=200)
        geo = compute_lego_geometry(landmarks)

        # 凸点的 y 坐标应在头部上方
        assert geo['stud_center'][1] < geo['center_y']

    def test_different_face_sizes_scale_proportionally(self):
        """不同大小的人脸应产生等比例缩放的结果"""
        small = compute_lego_geometry(self._make_landmarks(face_width=80))
        large = compute_lego_geometry(self._make_landmarks(face_width=160))

        ratio = large['head_width'] / small['head_width']
        assert 1.5 < ratio < 2.5  # 大约 2 倍


class TestDrawLegoHead:
    """测试在图像上绘制 LEGO 头部"""

    def _make_frame(self, width=640, height=480):
        """创建测试用空白图像"""
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _make_landmarks(self, center_x=320, center_y=240, face_width=120):
        """构造模拟特征点（包含个性化功能所需的所有点）"""
        landmarks = np.zeros((68, 2), dtype=int)
        half_w = face_width // 2

        # 下巴轮廓
        landmarks[0] = (center_x - half_w, center_y)
        landmarks[1] = (center_x - half_w + 5, center_y + 10)
        landmarks[2] = (center_x - half_w + 15, center_y + 20)
        landmarks[14] = (center_x + half_w - 15, center_y + 20)
        landmarks[15] = (center_x + half_w - 5, center_y + 10)
        landmarks[16] = (center_x + half_w, center_y)
        landmarks[8] = (center_x, center_y + half_w)

        # 眉毛
        landmarks[17] = (center_x - half_w + 10, center_y - 30)
        landmarks[21] = (center_x - 10, center_y - 30)
        landmarks[22] = (center_x + 10, center_y - 30)
        landmarks[26] = (center_x + half_w - 10, center_y - 30)
        landmarks[27] = (center_x, center_y - 20)
        landmarks[29] = (center_x, center_y - 5)

        # 左眼（含上下眼睑）
        landmarks[36] = (center_x - 30, center_y - 15)
        landmarks[37] = (center_x - 25, center_y - 20)
        landmarks[38] = (center_x - 15, center_y - 20)
        landmarks[39] = (center_x - 10, center_y - 15)
        landmarks[40] = (center_x - 15, center_y - 10)
        landmarks[41] = (center_x - 25, center_y - 10)

        # 右眼（含上下眼睑）
        landmarks[42] = (center_x + 10, center_y - 15)
        landmarks[43] = (center_x + 15, center_y - 20)
        landmarks[44] = (center_x + 25, center_y - 20)
        landmarks[45] = (center_x + 30, center_y - 15)
        landmarks[46] = (center_x + 25, center_y - 10)
        landmarks[47] = (center_x + 15, center_y - 10)

        # 嘴巴（含内唇）
        landmarks[48] = (center_x - 20, center_y + 15)
        landmarks[54] = (center_x + 20, center_y + 15)
        landmarks[51] = (center_x, center_y + 12)
        landmarks[57] = (center_x, center_y + 20)
        landmarks[62] = (center_x, center_y + 16)
        landmarks[66] = (center_x, center_y + 18)

        # 鼻子
        landmarks[30] = (center_x, center_y)
        landmarks[33] = (center_x, center_y + 5)

        return landmarks

    def test_returns_image_same_shape(self):
        """返回的图像应与输入尺寸相同"""
        frame = self._make_frame()
        landmarks = self._make_landmarks()
        result = draw_lego_head(frame, landmarks)

        assert result.shape == frame.shape

    def test_modifies_image(self):
        """应在图像上绘制内容（不是返回空白图像）"""
        frame = self._make_frame()
        landmarks = self._make_landmarks()
        result = draw_lego_head(frame.copy(), landmarks)

        assert not np.array_equal(result, frame)

    def test_head_region_contains_warm_color(self):
        """头部区域应包含暖色调像素（肤色与 LEGO 黄混合）"""
        frame = self._make_frame()
        landmarks = self._make_landmarks(center_x=320, center_y=240)
        result = draw_lego_head(frame, landmarks)

        # 检查中心区域是否包含非黑色的暖色像素（G 和 R 通道较高）
        region = result[200:280, 280:360]
        warm_mask = (region[:, :, 1] > 50) | (region[:, :, 2] > 50)
        warm_pixels = np.sum(warm_mask)
        assert warm_pixels > 100, f"头部区域应包含大量暖色像素，实际只有 {warm_pixels}"

    def test_eyes_are_black_dots(self):
        """眼睛位置应有黑色像素"""
        frame = self._make_frame(width=640, height=480)
        # 用白色填充以便检测黑色
        frame[:] = (255, 255, 255)
        landmarks = self._make_landmarks(center_x=320, center_y=240)
        result = draw_lego_head(frame, landmarks)

        geo = compute_lego_geometry(landmarks)
        lx, ly = geo['left_eye']
        rx, ry = geo['right_eye']

        # 眼睛位置附近应有深色像素
        left_region = result[max(0, ly - 5):ly + 5, max(0, lx - 5):lx + 5]
        right_region = result[max(0, ry - 5):ry + 5, max(0, rx - 5):rx + 5]

        assert np.min(left_region) < 50, "左眼位置应有深色像素"
        assert np.min(right_region) < 50, "右眼位置应有深色像素"

    def test_does_not_crash_on_edge_face(self):
        """人脸在图像边缘时不应崩溃"""
        frame = self._make_frame(width=640, height=480)
        landmarks = self._make_landmarks(center_x=50, center_y=50, face_width=80)
        # 不应抛出异常
        result = draw_lego_head(frame, landmarks)
        assert result.shape == frame.shape


class TestExtractFaceFeatures:
    """测试从真实图像和特征点提取个性化面部特征"""

    def _make_landmarks(self, center_x=320, center_y=240, face_width=120,
                        mouth_open=0, eye_close=0, brow_raise=0):
        """
        构造带有可调表情参数的模拟特征点。

        mouth_open: 0~1，嘴巴张开程度
        eye_close: 0~1，眼睛闭合程度
        brow_raise: -1~1，眉毛上扬/下压程度
        """
        landmarks = np.zeros((68, 2), dtype=int)
        half_w = face_width // 2

        landmarks[0] = (center_x - half_w, center_y)
        landmarks[1] = (center_x - half_w + 5, center_y + 10)
        landmarks[2] = (center_x - half_w + 15, center_y + 20)
        landmarks[14] = (center_x + half_w - 15, center_y + 20)
        landmarks[15] = (center_x + half_w - 5, center_y + 10)
        landmarks[16] = (center_x + half_w, center_y)
        landmarks[8] = (center_x, center_y + half_w)

        brow_offset = int(brow_raise * 15)
        # 眉毛内侧（靠近鼻梁）抬高更多，外侧抬高较少，形成角度变化
        landmarks[17] = (center_x - half_w + 10, center_y - 30)
        landmarks[19] = (center_x - 20, center_y - 35 - brow_offset)
        landmarks[21] = (center_x - 10, center_y - 30 - brow_offset)
        landmarks[22] = (center_x + 10, center_y - 30 - brow_offset)
        landmarks[24] = (center_x + 20, center_y - 35 - brow_offset)
        landmarks[26] = (center_x + half_w - 10, center_y - 30)
        landmarks[27] = (center_x, center_y - 20)
        landmarks[29] = (center_x, center_y - 5)

        eye_delta = int(eye_close * 5)
        landmarks[36] = (center_x - 30, center_y - 15)
        landmarks[39] = (center_x - 10, center_y - 15)
        landmarks[37] = (center_x - 25, center_y - 20 + eye_delta)
        landmarks[38] = (center_x - 15, center_y - 20 + eye_delta)
        landmarks[40] = (center_x - 15, center_y - 10 - eye_delta)
        landmarks[41] = (center_x - 25, center_y - 10 - eye_delta)

        landmarks[42] = (center_x + 10, center_y - 15)
        landmarks[45] = (center_x + 30, center_y - 15)
        landmarks[43] = (center_x + 15, center_y - 20 + eye_delta)
        landmarks[44] = (center_x + 25, center_y - 20 + eye_delta)
        landmarks[46] = (center_x + 25, center_y - 10 - eye_delta)
        landmarks[47] = (center_x + 15, center_y - 10 - eye_delta)

        mouth_delta = int(mouth_open * 15)
        landmarks[48] = (center_x - 20, center_y + 15)
        landmarks[54] = (center_x + 20, center_y + 15)
        landmarks[51] = (center_x, center_y + 12)
        landmarks[57] = (center_x, center_y + 20 + mouth_delta)
        landmarks[62] = (center_x, center_y + 14)
        landmarks[66] = (center_x, center_y + 18 + mouth_delta)

        landmarks[30] = (center_x, center_y)
        landmarks[33] = (center_x, center_y + 5)

        return landmarks

    def _make_face_frame(self, center_x=320, center_y=240, skin_bgr=(130, 180, 220),
                         hair_bgr=(30, 30, 50)):
        """创建模拟人脸的图像：肤色区域 + 头发区域"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 头发区域（额头上方）
        frame[max(0,center_y - 120):max(0,center_y - 30), max(0,center_x - 80):min(640,center_x + 80)] = hair_bgr
        # 肤色区域（脸部，覆盖所有采样点含脸颊）
        frame[max(0,center_y - 30):min(480,center_y + 80), max(0,center_x - 80):min(640,center_x + 80)] = skin_bgr
        return frame

    # --- 眼睛开合 ---
    def test_eye_openness_normal(self):
        """正常睁眼时开合度应较高"""
        landmarks = self._make_landmarks(eye_close=0)
        left_open, right_open = compute_eye_openness(landmarks)
        assert left_open > 0.5
        assert right_open > 0.5

    def test_eye_openness_closed(self):
        """闭眼时开合度应较低"""
        landmarks = self._make_landmarks(eye_close=0.9)
        left_open, right_open = compute_eye_openness(landmarks)
        assert left_open < 0.3
        assert right_open < 0.3

    def test_eye_openness_returns_values_0_to_1(self):
        """眼睛开合度应在 0~1 范围"""
        for close in [0, 0.3, 0.6, 0.9]:
            landmarks = self._make_landmarks(eye_close=close)
            left, right = compute_eye_openness(landmarks)
            assert 0 <= left <= 1.0
            assert 0 <= right <= 1.0

    # --- 嘴巴开合 ---
    def test_mouth_openness_closed(self):
        """闭嘴时开合度应较低"""
        landmarks = self._make_landmarks(mouth_open=0)
        openness = compute_mouth_openness(landmarks)
        assert openness < 0.3

    def test_mouth_openness_open(self):
        """张嘴时开合度应较高"""
        landmarks = self._make_landmarks(mouth_open=0.8)
        openness = compute_mouth_openness(landmarks)
        assert openness > 0.4

    # --- 眉毛角度 ---
    def test_brow_angles_neutral(self):
        """平静表情时两侧眉毛角度应接近对称"""
        landmarks = self._make_landmarks(brow_raise=0)
        left_angle, right_angle = compute_brow_angles(landmarks)
        assert abs(left_angle - right_angle) < 15  # 角度差小于 15 度

    def test_brow_angles_raised(self):
        """挑眉时眉毛角度应改变"""
        neutral = self._make_landmarks(brow_raise=0)
        raised = self._make_landmarks(brow_raise=0.8)
        _, n_right = compute_brow_angles(neutral)
        _, r_right = compute_brow_angles(raised)
        # 挑眉后角度应有明显变化
        assert abs(n_right - r_right) > 3

    # --- 肤色提取 ---
    def test_sample_skin_color_returns_bgr(self):
        """肤色提取应返回 (B, G, R) 三通道值"""
        frame = self._make_face_frame(skin_bgr=(130, 180, 220))
        landmarks = self._make_landmarks()
        color = sample_skin_color(frame, landmarks)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_sample_skin_color_matches_face(self):
        """提取的肤色应接近真实脸部颜色"""
        expected = (130, 180, 220)
        frame = self._make_face_frame(skin_bgr=expected)
        landmarks = self._make_landmarks()
        color = sample_skin_color(frame, landmarks)
        # 每个通道误差不超过 40
        for i in range(3):
            assert abs(int(color[i]) - expected[i]) < 40, \
                f"通道 {i}: 期望 {expected[i]}，实际 {color[i]}"

    # --- 头发颜色提取 ---
    def test_sample_hair_color_returns_bgr(self):
        """头发颜色提取应返回 (B, G, R) 三通道值"""
        frame = self._make_face_frame(hair_bgr=(30, 30, 50))
        landmarks = self._make_landmarks()
        color = sample_hair_color(frame, landmarks)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    # --- extract_face_features 综合 ---
    def test_extract_face_features_returns_all_keys(self):
        """综合提取应返回所有个性化特征"""
        frame = self._make_face_frame()
        landmarks = self._make_landmarks()
        features = extract_face_features(frame, landmarks)

        required = ['eye_openness_left', 'eye_openness_right',
                     'mouth_openness', 'brow_angle_left', 'brow_angle_right',
                     'skin_color', 'hair_color']
        for key in required:
            assert key in features, f"缺少键: {key}"

    # --- 个性化绘制 ---
    def test_draw_lego_head_with_open_mouth(self):
        """张嘴时 LEGO 头部的嘴巴区域应有更大的深色区域"""
        closed_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        open_frame = np.full((480, 640, 3), 200, dtype=np.uint8)

        closed_lm = self._make_landmarks(mouth_open=0)
        open_lm = self._make_landmarks(mouth_open=0.9)

        draw_lego_head(closed_frame, closed_lm)
        draw_lego_head(open_frame, open_lm)

        # 嘴巴区域：头部中心偏下
        mouth_region_closed = closed_frame[255:290, 290:350]
        mouth_region_open = open_frame[255:290, 290:350]

        dark_closed = np.sum(mouth_region_closed < 50)
        dark_open = np.sum(mouth_region_open < 50)

        assert dark_open > dark_closed, \
            f"张嘴时深色像素({dark_open})应多于闭嘴时({dark_closed})"

    def test_draw_lego_head_has_eyebrows(self):
        """LEGO 头部应有眉毛"""
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        landmarks = self._make_landmarks()
        result = draw_lego_head(frame, landmarks)

        geo = compute_lego_geometry(landmarks)
        # 眉毛应在眼睛上方
        brow_y = geo['left_eye'][1] - int(geo['head_width'] * 0.12)
        brow_region = result[max(0, brow_y - 5):brow_y + 5,
                             geo['left_eye'][0] - 10:geo['right_eye'][0] + 10]
        # 眉毛区域应有深色像素
        assert np.min(brow_region) < 80, "眉毛区域应有深色像素"
