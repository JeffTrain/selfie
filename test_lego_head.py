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

    def test_head_color_is_predominantly_lego_yellow(self):
        """头部颜色应以 LEGO 经典黄色为主调，而非肤色

        LEGO 小人仔最标志性的特征就是亮黄色头部。
        在 BGR 空间中，LEGO 黄色表现为 R>200, G>180, B<80。
        头部中心区域的平均颜色应满足此条件。
        """
        frame = self._make_frame()
        # 用肤色填充脸部区域（模拟真实场景采样到肤色）
        frame[200:280, 280:360] = (130, 180, 220)  # 典型肤色 BGR
        landmarks = self._make_landmarks(center_x=320, center_y=240)
        result = draw_lego_head(frame, landmarks)

        # 取头部正中心的色块（避免边缘/轮廓）
        region = result[225:255, 300:340]
        avg_b = np.mean(region[:, :, 0])
        avg_g = np.mean(region[:, :, 1])
        avg_r = np.mean(region[:, :, 2])

        # R 和 G 通道要高（黄色特征），B 通道要低
        assert avg_r > 200, f"R 通道 {avg_r:.0f} 太低，LEGO 黄色 R 应 > 200"
        assert avg_g > 170, f"G 通道 {avg_g:.0f} 太低，LEGO 黄色 G 应 > 170"
        assert avg_b < 100, f"B 通道 {avg_b:.0f} 太高，LEGO 黄色 B 应 < 100"

    def test_head_has_thick_outline(self):
        """LEGO 头部应有明显的深色轮廓线

        LEGO 小人仔的塑料感来自粗实的轮廓边缘。
        头部左右边缘应有连续的深色像素带。
        """
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        landmarks = self._make_landmarks(center_x=320, center_y=240)
        result = draw_lego_head(frame, landmarks)

        geo = compute_lego_geometry(landmarks)
        cx = geo['center_x']
        cy = geo['center_y']
        half_w = geo['head_width'] // 2

        # 检查左边缘附近是否有深色轮廓像素
        left_edge_x = cx - half_w
        edge_strip = result[cy - 20:cy + 20, max(0, left_edge_x - 2):left_edge_x + 5]
        min_brightness = np.min(np.mean(edge_strip, axis=2))
        assert min_brightness < 150, f"左边缘最暗像素 {min_brightness:.0f}，应有清晰轮廓 (< 150)"

    def test_hair_covers_top_of_head(self):
        """头发部分应覆盖头顶上方区域

        LEGO 发型像头盔一样扣在头上，覆盖头顶以上的区域。
        """
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        landmarks = self._make_landmarks(center_x=320, center_y=240)
        result = draw_lego_head(frame, landmarks)

        geo = compute_lego_geometry(landmarks)
        cx = geo['center_x']
        cy = geo['center_y']
        half_h = geo['head_height'] // 2

        # 头顶上方区域不应全是背景色，应被头发覆盖
        hair_region = result[max(0, cy - half_h - 20):cy - half_h + 5,
                             cx - 20:cx + 20]
        not_bg = np.sum(np.any(hair_region != 200, axis=2))
        assert not_bg > 50, f"头顶区域应被头发覆盖，非背景像素只有 {not_bg}"

    def test_stud_is_visible_above_head(self):
        """凸点应在头顶清晰可见

        LEGO 标志性的顶部圆柱凸点是识别 LEGO 头的关键。
        """
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)  # 深灰背景
        landmarks = self._make_landmarks(center_x=320, center_y=240)
        result = draw_lego_head(frame, landmarks)

        geo = compute_lego_geometry(landmarks)
        sx, sy = geo['stud_center']
        sr = geo['stud_radius']

        # 凸点位置应有非背景色的明亮像素
        stud_region = result[max(0, sy - sr):sy + sr,
                             max(0, sx - sr):sx + sr]
        bright_pixels = np.sum(np.mean(stud_region, axis=2) > 150)
        assert bright_pixels > 20, f"凸点位置应有明亮像素，实际只有 {bright_pixels}"

    def test_3d_cylinder_shading_center_brighter_than_edges(self):
        """3D 圆柱体效果：头部中心应比左右边缘更亮

        真实 LEGO 头是圆柱体，中心受到正面光照最亮，
        两侧因曲面而逐渐变暗，产生立体感。
        """
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        landmarks = self._make_landmarks(center_x=320, center_y=240)
        result = draw_lego_head(frame, landmarks)

        geo = compute_lego_geometry(landmarks)
        cx = geo['center_x']
        cy = geo['center_y']
        half_w = geo['head_width'] // 2

        # 取头部下方空白区域（嘴巴和底部椭圆之间，确保无遮挡）
        y_sample = cy + int(geo['head_height'] * 0.38)
        center_strip = result[y_sample - 3:y_sample + 3, cx - 10:cx + 10]
        left_strip = result[y_sample - 3:y_sample + 3,
                            max(0, cx - half_w + 8):cx - half_w + 18]
        right_strip = result[y_sample - 3:y_sample + 3,
                             cx + half_w - 18:cx + half_w - 8]

        center_brightness = np.mean(center_strip)
        left_brightness = np.mean(left_strip)
        right_brightness = np.mean(right_strip)

        assert center_brightness > left_brightness + 10, \
            f"中心({center_brightness:.0f})应比左侧({left_brightness:.0f})更亮"
        assert center_brightness > right_brightness + 10, \
            f"中心({center_brightness:.0f})应比右侧({right_brightness:.0f})更亮"

    def test_3d_head_has_elliptical_chin(self):
        """3D 圆柱体底部应有椭圆弧形下巴（不被头发遮挡）

        从正面看，圆柱体底部呈椭圆弧线收窄。
        底部椭圆弧最低点附近的宽度应明显窄于头部中段。
        这是区分 2D 矩形和 3D 圆柱的关键视觉特征。
        """
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        landmarks = self._make_landmarks(center_x=320, center_y=240)
        result = draw_lego_head(frame, landmarks)

        geo = compute_lego_geometry(landmarks)
        cx = geo['center_x']
        cy = geo['center_y']
        hw = geo['head_width']
        hh = geo['head_height']
        half_h = hh // 2
        cap_ry = int(hh * 0.08)  # 与渲染代码一致

        # 底部椭圆弧最低点附近 vs 中段
        bottom_y = cy + half_h + cap_ry - 3  # 接近椭圆弧底端
        mid_y = cy

        def count_non_bg_width(y_pos):
            row = result[y_pos, max(0, cx - hw):cx + hw]
            non_bg = np.any(row != 100, axis=1)
            if np.any(non_bg):
                indices = np.where(non_bg)[0]
                return indices[-1] - indices[0]
            return 0

        bottom_width = count_non_bg_width(bottom_y)
        mid_width = count_non_bg_width(mid_y)

        # 底部宽度应明显小于中段（椭圆收窄）
        assert bottom_width < mid_width * 0.8, \
            f"底部宽度({bottom_width})应窄于中段({mid_width})的80%，体现椭圆弧"

    def test_3d_bottom_chin_is_visible(self):
        """3D 圆柱体底部应有下巴弧线

        LEGO 头底部是椭圆弧形的下巴/颈部连接处。
        底部附近应有深色的弧线或阴影。
        """
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        landmarks = self._make_landmarks(center_x=320, center_y=240)
        result = draw_lego_head(frame, landmarks)

        geo = compute_lego_geometry(landmarks)
        cx = geo['center_x']
        cy = geo['center_y']
        half_h = geo['head_height'] // 2

        # 头部底部边缘附近应有明显区别于背景的像素
        bottom_y = cy + half_h
        bottom_region = result[bottom_y - 5:bottom_y + 5, cx - 20:cx + 20]
        # 底部应有非背景色的像素（头部颜色或阴影）
        non_bg_count = np.sum(np.any(bottom_region != 100, axis=2))
        assert non_bg_count > 30, f"底部应有可见像素，实际非背景像素 {non_bg_count}"


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
