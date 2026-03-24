import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import os
from pathlib import Path
import logging
import sys

# MediaPipe 导入

try:
    import mediapipe as mp

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions
    Image = mp.Image
    ImageFormat = mp.ImageFormat
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3]
)
logger = logging.getLogger(__name__)


class EnhancedEyeTracker:
    """
    增强版眼动追踪器 - 基于眼睑顶点距离的眨眼检测

    核心功能：
        1. 人脸检测：YOLOv8 快速定位人脸区域
        2. 眼睛定位：MediaPipe 468个3D关键点精确定位眼睛
        3. 眨眼检测：基于上下眼睑顶点距离，检测眨眼事件
        4. 视线估计：根据瞳孔在眼眶中的位置，映射到3x3屏幕网格

    可调参数：
        EYE_MARGIN: 眼睛区域扩展边距
        BLINK_HISTORY_SIZE: 眨眼检测历史帧数
        BLINK_THRESHOLD: 眨眼阈值（当前距离 < 历史平均 × 阈值）
        BLINK_COOLDOWN: 眨眼后冷却帧数，防止重复计数
        GAZE_HISTORY_SIZE: 视线历史帧数，用于平滑
        GAZE_SMOOTH_FRAMES: 视线平滑所需帧数
        DEBUG_MODE: 调试模式开关
    """

    # ==================== MediaPipe 关键点索引 ====================
    # 参考：MediaPipe Face Mesh 468个3D关键点
    # 官方文档：https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    # 左眼轮廓关键点（用于绘制眼睛区域边界框）
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144, 163, 7]
    # 索引说明：
    # 33  - 左眼内眼角下缘
    # 160 - 左眼上眼睑外侧
    # 158 - 左眼上眼睑中部
    # 133 - 左眼外眼角
    # 153 - 左眼下眼睑外侧
    # 144 - 左眼上眼睑内侧
    # 163 - 左眼下眼睑内侧
    # 7   - 左眼内眼角上缘

    # 右眼轮廓关键点（用于绘制眼睛区域边界框）
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380, 373, 362]
    # 索引说明：
    # 362 - 右眼内眼角下缘
    # 385 - 右眼上眼睑外侧
    # 387 - 右眼上眼睑中部
    # 263 - 右眼外眼角
    # 373 - 右眼下眼睑外侧
    # 380 - 右眼上眼睑内侧
    # 373 - 右眼下眼睑内侧
    # 362 - 右眼内眼角上缘

    # 瞳孔关键点（用于视线估计，取平均值提高精度）
    LEFT_PUPIL_INDICES = [468, 469, 470, 471]  # 左眼瞳孔周围4个点
    RIGHT_PUPIL_INDICES = [473, 474, 475, 476]  # 右眼瞳孔周围4个点

    # ==================== 眼睑顶点关键点（核心眨眼检测） ====================
    # 选择理由：这些点是上下眼睑在瞳孔正上方/下方的最高/最低点
    # 测量眼睑垂直距离，最能反映眼睛张开程度
    LEFT_UPPER_LID = 159  # 左上眼睑最高点（瞳孔正上方）
    LEFT_LOWER_LID = 145  # 左下眼睑最低点（瞳孔正下方）
    RIGHT_UPPER_LID = 386  # 右上眼睑最高点（瞳孔正上方）
    RIGHT_LOWER_LID = 374  # 右下眼睑最低点（瞳孔正下方）

    # ==================== 可调参数（可根据实际效果调整） ====================
    # 眼睛区域扩展边距（像素），增加上下文信息，提高检测稳定性
    EYE_MARGIN = 10

    # 眨眼检测参数
    BLINK_HISTORY_SIZE = 15  # 历史帧数，用于计算正常状态基准
    BLINK_THRESHOLD = 0.8  # 眨眼阈值：当前距离 < 历史平均 × 阈值时判定为眨眼
    BLINK_CONFIRM = 0.8  # 连续帧验证阈值：前一帧 < 历史平均 × 阈值，避免误检
    BLINK_COOLDOWN = 3  # 冷却帧数，眨眼后暂停检测，防止重复计数

    # 视线平滑参数
    GAZE_HISTORY_SIZE = 10  # 视线历史帧数，用于平滑
    GAZE_SMOOTH_FRAMES = 3  # 平滑所需的最小帧数

    # 调试参数
    DEBUG_MODE = False  # 调试模式，输出详细信息
    SHOW_LANDMARKS = True  # 是否显示眼睛轮廓关键点（青色点）
    SHOW_DISTANCE = True  # 是否显示眼睑距离数值

    # ==================== 视线灵敏度参数 ====================
    GAZE_SENSITIVITY_X = 3.0  # 水平灵敏度（左右）
    GAZE_SENSITIVITY_Y = 3.0  # 垂直灵敏度（上下）


    # 垂直灵敏度（上下）灵敏度系数，越大越灵敏（建议1.0-3.0）

    def __init__(self, model_name='yolov8n.pt', confidence=0.5):
        """
        初始化眼动追踪器

        Args:
            model_name: YOLO模型文件名，默认'yolov8n.pt'
            confidence: YOLO检测置信度阈值 (0-1)，默认0.5
        """
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe未安装，请执行: pip install mediapipe")
            raise ImportError("MediaPipe is required")

        # 模型参数
        self.confidence = confidence

        # 统计信息
        self.frame_count = 0  # 总处理帧数
        self.detection_count = 0  # 成功检测到人脸的帧数
        self.blink_count = 0  # 累计眨眼次数
        self.start_time = time.time()  # 启动时间

        # 状态控制
        self.blink_cooldown = 0  # 眨眼冷却计数器

        # 历史队列
        self.gaze_history = deque(maxlen=self.GAZE_HISTORY_SIZE)  # 视线历史
        self.blink_history = deque(maxlen=self.BLINK_HISTORY_SIZE)  # 眼睑距离历史

        # 屏幕区域划分（3x3网格，用于量化视线方向）
        self.screen_regions = {}
        for y in range(5):
            for x in range(5):
                idx = y * 5 + x
                # 方向描述
                x_desc = ["Far Left", "Left", "Center", "Right", "Far Right"][x]
                y_desc = ["Top", "Top-Mid", "Center", "Bottom-Mid", "Bottom"][y]
                self.screen_regions[idx] = f"{y_desc}{x_desc}"

        # 调试数据存储
        self.debug_info = {
            'left_dist': [],  # 左眼距离历史
            'right_dist': [],  # 右眼距离历史
            'blink_detected': False  # 当前帧是否检测到眨眼
        }

        # 初始化模型
        self._init_yolo(model_name)
        self._init_mediapipe()

        logger.info("=" * 60)
        logger.info("增强版眼动追踪系统初始化完成")
        logger.info(f"YOLO置信度: {confidence}")
        logger.info(f"眨眼检测参数: 阈值={self.BLINK_THRESHOLD}, 冷却={self.BLINK_COOLDOWN}帧")
        logger.info(f"眼睑关键点: 左眼({self.LEFT_UPPER_LID},{self.LEFT_LOWER_LID})")
        logger.info(f"           右眼({self.RIGHT_UPPER_LID},{self.RIGHT_LOWER_LID})")
        logger.info("=" * 60)

    # ==================== 初始化方法 ====================
    def _init_yolo(self, model_name: str):
        """
        初始化YOLO模型

        Args:
            model_name: YOLO模型文件名
        """
        model_path = Path(model_name)
        if not model_path.exists():
            logger.warning(f"本地未找到模型: {model_name}")

        try:
            logger.info(f"加载YOLO模型: {model_name}")
            self.model = YOLO(model_name)
            logger.info("YOLO模型加载成功")
        except Exception as e:
            logger.error(f"YOLO加载失败: {e}")
            raise

    def _init_mediapipe(self):
        """初始化MediaPipe Face Landmarker"""
        self.detector = None
        model_path = "face_landmarker.task"

        if not os.path.exists(model_path):
            logger.info("下载MediaPipe模型...")
            self._download_mediapipe_model(model_path)
        else:
            logger.info(f"模型文件已存在: {model_path}")

        try:
            base_options = BaseOptions(model_asset_path=model_path)
            options = FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,  # 检测1张人脸
                min_face_detection_confidence=0.3,  # 最小检测置信度
                min_face_presence_confidence=0.3,  # 最小存在置信度
                min_tracking_confidence=0.3,  # 最小跟踪置信度
                output_face_blendshapes=False,  # 不输出表情混合形状（节省资源）
                output_facial_transformation_matrixes=False  # 不输出变换矩阵（节省资源）
            )
            self.detector = FaceLandmarker.create_from_options(options)
            logger.info("FaceLandmarker初始化成功")
        except Exception as e:
            logger.error(f"MediaPipe初始化失败: {e}")
            raise

    def _download_mediapipe_model(self, model_path: str):
        """
        下载MediaPipe模型文件

        Args:
            model_path: 模型保存路径
        """
        import urllib.request
        urls = [
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            "https://github.com/google/mediapipe/raw/master/mediapipe/modules/face_landmarker/face_landmarker.task"
        ]
        for idx, url in enumerate(urls, 1):
            try:
                logger.info(f"从镜像源 {idx} 下载...")
                urllib.request.urlretrieve(url, model_path)
                logger.info(f"模型下载成功")
                return
            except:
                continue
        raise RuntimeError("模型下载失败")

    # ==================== 人脸检测 ====================
    def detect_face(self, frame):
        """
        使用YOLO检测人脸区域

        Args:
            frame: 输入图像帧（BGR格式）

        Returns:
            tuple: (x1, y1, x2, y2) 人脸边界框坐标，未检测到返回None
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                if class_name == 'person' or class_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    return (x1, y1, x2, y2)
        return None

    # ==================== 眼睛提取 ====================
    def extract_eyes_mediapipe(self, frame, face_bbox):
        """
        使用MediaPipe精确提取眼睛区域

        Args:
            frame: 原始图像帧
            face_bbox: 人脸边界框 (x1, y1, x2, y2)

        Returns:
            tuple: (left_eye_data, right_eye_data) 左右眼数据字典
        """
        if self.detector is None:
            return None, None

        x1, y1, x2, y2 = face_bbox
        face_roi = frame[y1:y2, x1:x2]

        if face_roi.size == 0:
            return None, None

        try:
            rgb_frame = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            if not detection_result.face_landmarks:
                return None, None

            face_landmarks = detection_result.face_landmarks[0]
            h, w = face_roi.shape[:2]

            left_eye_data = self._extract_eye_data(
                frame, face_roi, face_landmarks,
                self.LEFT_EYE_INDICES, self.LEFT_PUPIL_INDICES,
                x1, y1, w, h, "left",
                self.LEFT_UPPER_LID, self.LEFT_LOWER_LID
            )

            right_eye_data = self._extract_eye_data(
                frame, face_roi, face_landmarks,
                self.RIGHT_EYE_INDICES, self.RIGHT_PUPIL_INDICES,
                x1, y1, w, h, "right",
                self.RIGHT_UPPER_LID, self.RIGHT_LOWER_LID
            )

            if self.DEBUG_MODE and left_eye_data and right_eye_data:
                logger.debug(f"左眼距离: {left_eye_data['eyelid_distance']:.1f}px, "
                             f"右眼距离: {right_eye_data['eyelid_distance']:.1f}px")

            return left_eye_data, right_eye_data

        except Exception as e:
            logger.debug(f"MediaPipe处理失败: {e}")
            return None, None

    def _extract_eye_data(self, frame, face_roi, face_landmarks,
                          eye_indices, pupil_indices, offset_x, offset_y,
                          roi_w, roi_h, side, upper_lid_idx, lower_lid_idx):
        """
        提取单眼数据

        Args:
            frame: 原始图像帧
            face_roi: 人脸区域图像
            face_landmarks: MediaPipe检测到的面部关键点
            eye_indices: 眼睛轮廓关键点索引列表
            pupil_indices: 瞳孔关键点索引列表
            offset_x, offset_y: 人脸ROI在原图中的偏移量
            roi_w, roi_h: 人脸ROI的宽高
            side: 眼睛侧别 ("left" 或 "right")
            upper_lid_idx: 上眼睑关键点索引
            lower_lid_idx: 下眼睑关键点索引

        Returns:
            dict: 包含眼睛所有信息的字典
                - side: 眼睛侧别
                - region: 眼睛区域图像
                - bbox: 边界框 (x1, y1, x2, y2)
                - landmarks: 眼睛轮廓关键点坐标
                - pupil: 瞳孔中心坐标
                - eyelid_distance: 上下眼睑距离（像素）
                - upper_y: 上眼睑Y坐标
                - lower_y: 下眼睑Y坐标
        """
        # 获取眼睛轮廓点（用于边界框）
        eye_points = []
        for idx in eye_indices:
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                x = int(lm.x * roi_w)
                y = int(lm.y * roi_h)
                eye_points.append([x, y])

        if not eye_points:
            return None

        eye_points = np.array(eye_points)

        # 获取瞳孔点（用于视线估计）
        pupil_points = []
        for idx in pupil_indices:
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                x = int(lm.x * roi_w)
                y = int(lm.y * roi_h)
                pupil_points.append([x, y])

        # 获取眼睑顶点坐标（核心：眨眼检测）
        upper_lm = face_landmarks[upper_lid_idx]
        lower_lm = face_landmarks[lower_lid_idx]

        upper_y = upper_lm.y * roi_h
        lower_y = lower_lm.y * roi_h
        eyelid_distance = lower_y - upper_y  # 眼睑垂直距离（像素）

        # 计算眼睛边界框
        min_x = int(np.min(eye_points[:, 0]) - self.EYE_MARGIN)
        max_x = int(np.max(eye_points[:, 0]) + self.EYE_MARGIN)
        min_y = int(np.min(eye_points[:, 1]) - self.EYE_MARGIN)
        max_y = int(np.max(eye_points[:, 1]) + self.EYE_MARGIN)

        min_x = max(0, min_x)
        max_x = min(roi_w, max_x)
        min_y = max(0, min_y)
        max_y = min(roi_h, max_y)

        # 全局坐标（相对原图）
        global_x1 = offset_x + min_x
        global_y1 = offset_y + min_y
        global_x2 = offset_x + max_x
        global_y2 = offset_y + max_y

        eye_img = frame[global_y1:global_y2, global_x1:global_x2]

        # 瞳孔中心（全局坐标）
        pupil_center = None
        if pupil_points:
            pupil_center = np.mean(pupil_points, axis=0)
            pupil_center = (offset_x + pupil_center[0], offset_y + pupil_center[1])

        return {
            'side': side,
            'region': eye_img,
            'bbox': (global_x1, global_y1, global_x2, global_y2),
            'landmarks': eye_points + np.array([offset_x, offset_y]),
            'pupil': pupil_center,
            'eyelid_distance': eyelid_distance,
            'upper_y': upper_y + offset_y,
            'lower_y': lower_y + offset_y
        }

    # ==================== 眨眼检测 ====================
    def detect_blink(self, left_eye_data, right_eye_data):
        """
        基于眼睑顶点距离的眨眼检测算法

        算法原理：
            1. 计算双眼上下眼睑顶点的垂直距离（像素）
            2. 保存最近15帧的距离数据
            3. 计算历史平均距离作为正常状态基准
            4. 当当前距离 < 历史平均 × 阈值时，判定为眨眼
            5. 冷却机制防止单次眨眼被重复计数

        Args:
            left_eye_data: 左眼数据字典
            right_eye_data: 右眼数据字典

        Returns:
            bool: 是否检测到眨眼
        """
        if left_eye_data is None or right_eye_data is None:
            return False

        # 获取眼睑距离（像素）
        left_dist = left_eye_data.get('eyelid_distance', 0)
        right_dist = right_eye_data.get('eyelid_distance', 0)

        # 无效数据检查
        if left_dist <= 0 or right_dist <= 0:
            return False

        avg_dist = (left_dist + right_dist) / 2
        self.blink_history.append(avg_dist)

        # 保持队列长度
        while len(self.blink_history) > self.BLINK_HISTORY_SIZE:
            self.blink_history.popleft()

        # 冷却机制：眨眼后短时间内不重复检测
        if self.blink_cooldown > 0:
            self.blink_cooldown -= 1
            return False

        # 需要足够的历史数据（至少10帧）
        if len(self.blink_history) >= self.BLINK_HISTORY_SIZE - 5:
            # 计算历史平均（排除最近2帧，避免受当前眨眼影响）
            history_vals = list(self.blink_history)[:-2]
            history_avg = np.mean(history_vals)

            # 调试输出
            if self.DEBUG_MODE:
                logger.debug(f"距离: 当前={avg_dist:.1f}, 历史平均={history_avg:.1f}, "
                             f"阈值={history_avg * self.BLINK_THRESHOLD:.1f}")

            # 眨眼判断：当前距离 < 历史平均 × 阈值
            if history_avg > 0 and avg_dist < history_avg * self.BLINK_THRESHOLD:
                # 验证连续两帧都低（避免单帧噪声）
                if len(self.blink_history) >= 2:
                    prev_dist = self.blink_history[-2]
                    if prev_dist < history_avg * self.BLINK_CONFIRM:
                        # 设置冷却，避免连续触发
                        self.blink_cooldown = self.BLINK_COOLDOWN
                        if self.DEBUG_MODE:
                            logger.info(f"检测到眨眼! 距离={avg_dist:.1f}px")
                        return True


    # ==================== 眼线检测 ====================

    def estimate_gaze_region(self, left_eye_data, right_eye_data):
        """
        根据瞳孔位置估计视线方向（8方向）

        Returns:
            str: 视线方向 "Up", "Down", "Left", "Right", "UpLeft", "UpRight", "DownLeft", "DownRight", "Center"
        """
        if left_eye_data is None or right_eye_data is None:
            return None

        left_pupil = left_eye_data.get('pupil')
        right_pupil = right_eye_data.get('pupil')

        if left_pupil is None or right_pupil is None:
            return None

        left_bbox = left_eye_data['bbox']
        right_bbox = right_eye_data['bbox']

        if left_bbox[2] == left_bbox[0] or right_bbox[2] == right_bbox[0]:
            return None

        # 归一化坐标
        left_norm_x = (left_pupil[0] - left_bbox[0]) / (left_bbox[2] - left_bbox[0])
        left_norm_y = (left_pupil[1] - left_bbox[1]) / (left_bbox[3] - left_bbox[1])
        right_norm_x = (right_pupil[0] - right_bbox[0]) / (right_bbox[2] - right_bbox[0])
        right_norm_y = (right_pupil[1] - right_bbox[1]) / (right_bbox[3] - right_bbox[1])

        # 双眼平均
        avg_x = (left_norm_x + right_norm_x) / 2
        avg_y = (left_norm_y + right_norm_y) / 2

        # ===================计算偏移量===================
        offset_x = (avg_x - 0.5) * self.GAZE_SENSITIVITY_X
        offset_y = (avg_y - 0.45) * self.GAZE_SENSITIVITY_Y

        # 限制范围
        offset_x = max(-0.5, min(0.5, offset_x))
        offset_y = max(-0.5, min(0.5, offset_y))

        # 判断方向
        threshold = 0.15

        if offset_x > threshold:
            h_dir = "Right"
        elif offset_x < -threshold:
            h_dir = "Left"
        else:
            h_dir = ""

        if offset_y > threshold :
            v_dir = "Down"
        elif offset_y < -threshold :
            v_dir = "Up"
        else:
            v_dir = ""

        # 组合方向
        if v_dir and h_dir:
            direction = v_dir + h_dir
        elif v_dir:
            direction = v_dir
        elif h_dir:
            direction = h_dir
        else:
            direction = "Center"

        return direction


    # ==================== UI绘制 ====================
    def draw_interface(self, frame, face_bbox, left_eye_data, right_eye_data,
                       gaze_direction, is_blink):

        """
        绘制UI界面

        Args:
            frame: 图像帧
            face_bbox: 人脸边界框
            left_eye_data: 左眼数据
            right_eye_data: 右眼数据
            gaze_region: 视线区域ID
            is_blink: 是否眨眼

        Returns:
            np.ndarray: 绘制后的图像帧
        """
        # 绘制人脸框（绿色）
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制眼睛区域
        for eye_data in [left_eye_data, right_eye_data]:
            if eye_data and eye_data.get('bbox'):
                bbox = eye_data['bbox']

                # 眼睛区域框（蓝色）
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)

                # 绘制眼睛轮廓关键点（青色）
                if self.SHOW_LANDMARKS and eye_data.get('landmarks') is not None:
                    landmarks = eye_data['landmarks']
                    if isinstance(landmarks, np.ndarray):
                        for point in landmarks:
                            cv2.circle(frame, tuple(point.astype(int)), 1, (0, 255, 255), -1)

                # 绘制瞳孔（红色）
                pupil = eye_data.get('pupil')
                if pupil:
                    cv2.circle(frame, (int(pupil[0]), int(pupil[1])), 4, (0, 0, 255), -1)

                # 显示眼睑距离（黄色文字）
                if self.SHOW_DISTANCE:
                    dist = eye_data.get('eyelid_distance', 0)
                    cv2.putText(frame, f'{eye_data["side"][0]}_dist: {dist:.1f}px',
                                (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 显示视线区域（黄色文字）
        if gaze_direction is not None:
            cv2.putText(frame, f'Gaze: {gaze_direction}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 显示眨眼提示（红色文字）
        if is_blink:
            cv2.putText(frame, 'BLINK!', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示统计信息
        if self.frame_count > 0:
            detection_rate = (self.detection_count / self.frame_count) * 100
            cv2.putText(frame, f'Detection: {detection_rate:.1f}%', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f'Blinks: {self.blink_count}', (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 调试模式显示当前眼睑距离
            if self.DEBUG_MODE and left_eye_data and right_eye_data:
                left_dist = left_eye_data.get('eyelid_distance', 0)
                right_dist = right_eye_data.get('eyelid_distance', 0)
                cv2.putText(frame, f'L/R: {left_dist:.0f}/{right_dist:.0f}px', (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 使用说明
        cv2.putText(frame, 'q: quit | r: reset', (frame.shape[1] - 200, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    # ==================== 主循环 ====================
    def run(self):
        """
        主循环：打开摄像头并开始追踪

        工作流程：
            1. 初始化摄像头
            2. 循环读取视频帧
            3. 人脸检测 → 眼睛定位 → 视线估计 → 眨眼检测
            4. 实时可视化
            5. 处理键盘输入（q退出，r重置，d调试）
        """
        logger.info("初始化摄像头...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("无法打开摄像头")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"摄像头: {width}x{height}")
        logger.info("按 'q' 退出，按 'r' 重置，按 'd' 切换调试模式")
        logger.info("=" * 60)

        fps_start = time.time()
        fps_count = 0
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 水平翻转（镜像效果，让用户感觉更自然）
            frame = cv2.flip(frame, 1)
            self.frame_count += 1

            # 1. 人脸检测
            face_bbox = self.detect_face(frame)

            left_eye = right_eye = None
            gaze_region = None
            is_blink = False

            # 2. 眼睛定位和分析
            if face_bbox:
                self.detection_count += 1
                left_eye, right_eye = self.extract_eyes_mediapipe(frame, face_bbox)

                if left_eye and right_eye:
                    # 3. 视线估计
                    gaze_region = self.estimate_gaze_region(left_eye, right_eye)
                    # if gaze_region is not None:
                    #     self.gaze_history.append(gaze_region)
                    #     # 平滑处理：取历史中出现频率最高的区域
                    #     if len(self.gaze_history) >= self.GAZE_SMOOTH_FRAMES:
                    #         from collections import Counter
                    #         gaze_region = Counter(self.gaze_history).most_common(1)[0][0]

                    # 4. 眨眼检测
                    if self.detect_blink(left_eye, right_eye):
                        is_blink = True
                        self.blink_count += 1

            # 5. 绘制UI
            frame = self.draw_interface(frame, face_bbox, left_eye, right_eye,
                                        gaze_region, is_blink)

            # 6. FPS计算
            fps_count += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_count
                fps_count = 0
                fps_start = time.time()
            cv2.putText(frame, f'FPS: {fps}', (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 7. 显示图像
            cv2.imshow('Eye Tracking', frame)

            # 8. 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.frame_count = self.detection_count = self.blink_count = 0
                self.gaze_history.clear()
                self.blink_history.clear()
                self.blink_cooldown = 0
                logger.info("统计已重置")
            elif key == ord('d'):
                self.DEBUG_MODE = not self.DEBUG_MODE
                logger.info(f"调试模式: {'开启' if self.DEBUG_MODE else '关闭'}")

        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        if self.detector:
            self.detector.close()

        # 最终统计
        logger.info("=" * 60)
        logger.info(f"总帧数: {self.frame_count}")
        if self.frame_count > 0:
            logger.info(f"检测率: {self.detection_count / self.frame_count * 100:.1f}%")
        logger.info(f"眨眼次数: {self.blink_count}")
        logger.info("系统关闭")


def main():
    """主函数：程序入口"""
    logger.info("=" * 60)
    logger.info("增强版眼动追踪系统")
    logger.info("基于眼睑顶点距离的眨眼检测")
    logger.info("=" * 60)

    # 打印依赖库版本信息
    try:
        import mediapipe as mp
        logger.info(f"MediaPipe: {mp.__version__}")
    except:
        pass
    logger.info(f"OpenCV: {cv2.__version__}")
    logger.info(f"NumPy: {np.__version__}")

    try:
        tracker = EnhancedEyeTracker(model_name='yolov8n.pt', confidence=0.5)
        tracker.run()
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()