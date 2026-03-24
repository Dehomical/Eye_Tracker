import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
import sys
import logging

# MediaPipe 新版 Tasks API 导入
# MediaPipe 0.10.33 版本的正确导入方式

try:
    # 备用导入方式
    from mediapipe.tasks.python.vision.core import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.vision import Image, ImageFormat
except ImportError:
        # 如果都失败，尝试直接从 mediapipe 导入
    import mediapipe as mp

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions
    Image = mp.Image
    ImageFormat = mp.ImageFormat

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EyeTrackingSystem:
    """
    眼动追踪系统类 (使用 MediaPipe Tasks API)
    """

    # 定义眼睛关键点索引
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144, 163, 7]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380, 373, 362]

    # 瞳孔中心估计点
    LEFT_PUPIL = [468, 469, 470, 471]
    RIGHT_PUPIL = [473, 474, 475, 476]

    def __init__(self, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        """
        初始化眼动追踪系统
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # 模型文件路径
        self.model_path = self._download_model()

        try:
            # 创建 FaceLandmarkerOptions 配置对象
            base_options = BaseOptions(model_asset_path=self.model_path)

            options = FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )

            # 创建 FaceLandmarker 检测器
            self.detector = FaceLandmarker.create_from_options(options)
            logger.info("FaceLandmarker 初始化成功")

        except Exception as e:
            logger.error(f"初始化 FaceLandmarker 失败: {str(e)}")
            logger.info("尝试使用备用方案...")
            self.detector = None

        # 性能统计
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()

        logger.info("眼动追踪系统初始化完成")

    def _download_model(self) -> str:
        """
        下载MediaPipe模型文件
        """
        import os
        import urllib.request

        model_filename = "face_landmarker.task"

        # 检查模型文件是否存在
        if os.path.exists(model_filename):
            logger.info(f"模型文件已存在: {model_filename}")
            return model_filename

        # 下载模型文件
        logger.info("正在下载MediaPipe模型文件...")
        model_urls = [
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            "https://github.com/google/mediapipe/raw/master/mediapipe/modules/face_landmarker/face_landmarker.task"
        ]

        for url in model_urls:
            try:
                logger.info(f"尝试从 {url} 下载...")
                urllib.request.urlretrieve(url, model_filename)
                logger.info(f"模型下载成功: {model_filename}")
                return model_filename
            except Exception as e:
                logger.warning(f"下载失败: {e}")
                continue

        logger.error("无法下载模型文件，使用默认路径")
        return model_filename

    def calculate_iris_position(self, landmarks: List, eye_indices: List,
                                iris_indices: List, image_shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        计算虹膜/瞳孔位置相对于眼窝的偏移量
        """
        height, width = image_shape[:2]

        # 获取眼睛关键点坐标
        eye_points = []
        for i in eye_indices:
            if i < len(landmarks):
                eye_points.append([landmarks[i].x * width, landmarks[i].y * height])

        if not eye_points:
            return 0.0, 0.0

        eye_points = np.array(eye_points, dtype=np.float32)

        # 计算眼窝的边界
        x_min, y_min = eye_points.min(axis=0)
        x_max, y_max = eye_points.max(axis=0)

        # 计算眼窝中心
        eye_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)

        # 获取虹膜/瞳孔关键点
        iris_points = []
        for i in iris_indices:
            if i < len(landmarks):
                iris_points.append([landmarks[i].x * width, landmarks[i].y * height])

        if not iris_points:
            return 0.0, 0.0

        iris_points = np.array(iris_points, dtype=np.float32)

        # 计算虹膜中心
        iris_center = iris_points.mean(axis=0)

        # 计算归一化偏移量
        eye_width = x_max - x_min
        eye_height = y_max - y_min

        if eye_width > 0 and eye_height > 0:
            x_ratio = (iris_center[0] - eye_center[0]) / (eye_width / 2)
            y_ratio = (iris_center[1] - eye_center[1]) / (eye_height / 2)
        else:
            x_ratio = y_ratio = 0.0

        # 限制范围在[-1, 1]
        x_ratio = max(-1.0, min(1.0, x_ratio))
        y_ratio = max(-1.0, min(1.0, y_ratio))

        return x_ratio, y_ratio

    def draw_landmarks(self, image: np.ndarray, landmarks: List):
        """
        绘制面部关键点
        """
        height, width = image.shape[:2]

        # 绘制左眼轮廓
        for idx in self.LEFT_EYE_INDICES:
            if idx < len(landmarks):
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # 绘制右眼轮廓
        for idx in self.RIGHT_EYE_INDICES:
            if idx < len(landmarks):
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # 绘制左瞳孔
        for idx in self.LEFT_PUPIL:
            if idx < len(landmarks):
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

        # 绘制右瞳孔
        for idx in self.RIGHT_PUPIL:
            if idx < len(landmarks):
                point = landmarks[idx]
                x = int(point.x * width)
                y = int(point.y * height)
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

    def draw_eye_gaze(self, image: np.ndarray, landmarks: List,
                      left_gaze: Tuple[float, float], right_gaze: Tuple[float, float]):
        """
        在图像上绘制视线方向指示器
        """
        height, width = image.shape[:2]

        # 左眼中心
        if 33 < len(landmarks) and 133 < len(landmarks):
            left_eye_center = np.array([
                (landmarks[33].x + landmarks[133].x) / 2 * width,
                (landmarks[33].y + landmarks[133].y) / 2 * height
            ])

            # 绘制左眼视线箭头
            arrow_length = 50
            end_point = left_eye_center + np.array([left_gaze[0], left_gaze[1]]) * arrow_length
            cv2.arrowedLine(image,
                            left_eye_center.astype(int),
                            end_point.astype(int),
                            (0, 255, 0), 2)

        # 右眼中心
        if 362 < len(landmarks) and 263 < len(landmarks):
            right_eye_center = np.array([
                (landmarks[362].x + landmarks[263].x) / 2 * width,
                (landmarks[362].y + landmarks[263].y) / 2 * height
            ])

            # 绘制右眼视线箭头
            arrow_length = 50
            end_point = right_eye_center + np.array([right_gaze[0], right_gaze[1]]) * arrow_length
            cv2.arrowedLine(image,
                            right_eye_center.astype(int),
                            end_point.astype(int),
                            (0, 255, 0), 2)

    def process_frame(self, frame: np.ndarray) -> Tuple[
        np.ndarray, Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        处理视频帧，检测面部并计算眼动信息
        """
        if self.detector is None:
            return frame, None, None

        try:
            # 水平翻转图像
            frame = cv2.flip(frame, 1)

            # 转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 创建 MediaPipe Image 对象
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

            # 进行检测
            detection_result = self.detector.detect(mp_image)

            annotated_frame = frame.copy()
            left_gaze = None
            right_gaze = None

            # 处理检测结果
            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    # 绘制关键点
                    self.draw_landmarks(annotated_frame, face_landmarks)

                    # 计算眼睛虹膜位置
                    left_gaze = self.calculate_iris_position(
                        face_landmarks,
                        self.LEFT_EYE_INDICES,
                        self.LEFT_PUPIL,
                        annotated_frame.shape
                    )

                    right_gaze = self.calculate_iris_position(
                        face_landmarks,
                        self.RIGHT_EYE_INDICES,
                        self.RIGHT_PUPIL,
                        annotated_frame.shape
                    )

                    # 绘制视线方向
                    self.draw_eye_gaze(annotated_frame, face_landmarks,
                                       left_gaze, right_gaze)

            # 更新FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()

            # 显示FPS
            cv2.putText(annotated_frame, f'FPS: {self.fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示视线方向
            if left_gaze is not None and right_gaze is not None:
                avg_gaze_x = (left_gaze[0] + right_gaze[0]) / 2
                avg_gaze_y = (left_gaze[1] + right_gaze[1]) / 2

                # 判断方向
                direction = ""
                if abs(avg_gaze_x) > 0.2 or abs(avg_gaze_y) > 0.2:
                    if avg_gaze_x > 0.2:
                        direction += "Right"
                    elif avg_gaze_x < -0.2:
                        direction += "Left"

                    if avg_gaze_y > 0.2:
                        direction += "Dowm"
                    elif avg_gaze_y < -0.2:
                        direction += "Up"

                    if direction:
                        cv2.putText(annotated_frame, f'Direction: {direction}', (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 显示具体数值
                cv2.putText(annotated_frame, f'Gaze X: {avg_gaze_x:.2f}', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(annotated_frame, f'Gaze Y: {avg_gaze_y:.2f}', (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            return annotated_frame, left_gaze, right_gaze

        except Exception as e:
            logger.error(f"处理帧时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame, None, None

    def run(self, camera_index: int = 0):
        """
        运行眼动追踪系统主循环
        """
        try:
            logger.info("正在初始化摄像头...")
            cap = cv2.VideoCapture(camera_index)

            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 (索引: {camera_index})")

            # 设置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680) # 原 1280
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 原 720

            logger.info("眼动追踪系统启动成功，按ESC或q键退出")
            logger.info("提示: 请确保面部在摄像头视野内，光线充足")

            while True:
                ret, frame = cap.read()

                if not ret:
                    logger.warning("无法从摄像头读取帧")
                    time.sleep(0.1)
                    continue

                # 处理帧
                annotated_frame, left_gaze, right_gaze = self.process_frame(frame)

                # 显示结果
                cv2.imshow('Eye Tracking System', annotated_frame)

                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC或q
                    logger.info("用户请求退出")
                    break

        except Exception as e:
            logger.error(f"运行时错误: {str(e)}")
            raise

        finally:
            # 清理资源
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            if self.detector:
                self.detector.close()
            logger.info("系统已关闭")


def main():
    """
    主函数
    """
    logger.info("=" * 60)
    logger.info("眼动追踪系统 (Eye Tracking System)")
    logger.info("基于 MediaPipe Tasks API 开发")
    logger.info("=" * 60)

    # 打印系统信息
    try:
        import mediapipe as mp
        logger.info(f"MediaPipe版本: {mp.__version__}")
    except:
        logger.info("MediaPipe版本: 未检测到")

    logger.info(f"OpenCV版本: {cv2.__version__}")
    logger.info(f"NumPy版本: {np.__version__}")

    # 运行主程序
    try:
        tracker = EyeTrackingSystem(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        tracker.run(camera_index=0)
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序发生未预期的错误: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()