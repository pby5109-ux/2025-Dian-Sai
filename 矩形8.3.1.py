# K230无屏矩形定位：启动自动估计Otsu阈值，GPIO32按键重新估计。
import time, os, gc, sys,math
from math import atan,sqrt,atan2,sqrt
from media.sensor import *
from media.media import *
import cv_lite
from time import ticks_ms
from machine import UART
from machine import FPIOA
from machine import Pin

# 实例化FPIOA
fpioa = FPIOA()
# UART2和重新估计按键
fpioa.set_function(11, fpioa.UART2_TXD)
fpioa.set_function(12, fpioa.UART2_RXD)
fpioa.set_function(32, FPIOA.GPIO32)

RECALIBRATE_KEY = Pin(32, Pin.IN, Pin.PULL_DOWN)  # 高有效：未按为低，按下接3.3V，按下重新估计阈值

# UART2: baudrate 115200, 8bits, parity none, one stopbits
uart = UART(UART.UART2, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

DETECT_WIDTH = 480
DETECT_HEIGHT = 320
sensor = None
media_started = False
image_shape = [DETECT_HEIGHT, DETECT_WIDTH]  # cv_lite: 高、宽
S_THRESHOLD = 2000  # 候选最小面积，与灰度分割阈值不同
# -------------------------------
# 可调参数（建议调试时调整）/ Adjustable parameters (recommended for tuning)
# -------------------------------
canny_thresh1       = 50        # Canny 边缘检测低阈值 / Canny edge low threshold
canny_thresh2       = 150       # Canny 边缘检测高阈值 / Canny edge high threshold
approx_epsilon      = 0.04      # 多边形拟合精度（比例） / Polygon approximation precision (ratio)
area_min_ratio      = 0.001     # 最小面积比例（0~1） / Minimum area ratio (0~1)
max_angle_cos       = 0.5       # 最大角余弦（值越小越接近矩形） / Max cosine of angle (smaller closer to rectangle)
gaussian_blur_size  = 5         # 高斯模糊核大小（奇数） / Gaussian blur kernel size (odd number)
length_threshold=120
last=0


class DebouncedPress:
    """高有效按键（未按0、按下1）：状态稳定30ms后确认，长按只产生一次事件。"""
    def __init__(self, raw, now):
        self.raw = raw
        self.stable = raw
        self.changed = now

    def update(self, raw, now):
        if raw != self.raw:
            self.raw = raw
            self.changed = now
        if raw != self.stable and time.ticks_diff(now, self.changed) >= 30:
            self.stable = raw
            return raw == 1
        return False


class AutoThreshold:
    """启动或按键请求时自动估计，随后保持；不保存到SD卡。

    无灰度对比的帧不能用于标定，此时保留重估请求、跳过识别，
    下一帧重试，不用失效旧值冒充本次标定结果。
    """
    def __init__(self):
        self.value = None
        self.pending = True
        self.generation = 0

    def request(self):
        self.pending = True

    def get(self, gray):
        if self.pending:
            histogram = gray.get_histogram()
            stats = histogram.get_statistics()
            if stats.max() <= stats.min():
                return None
            value = int(histogram.get_threshold().value())
            if not 0 <= value <= 255:
                raise ValueError('Otsu threshold outside grayscale range')
            self.value = value
            self.pending = False
            self.generation += 1
            print('Otsu threshold:', self.value)
        return self.value


def prepare_detection(raw, control):
    gray = raw.to_grayscale()
    value = control.get(gray)
    if value is None:
        return None
    # 灰度二值图转回RGB888以保留现有cv_lite接口；坐标仍为480x320。
    return gray.binary([(0, value)]).to_rgb888()


def format_coord(coord):
    # 格式化为：符号位（+/-） + 3 位数值（补零）
    return f"{coord:+04d}"  # 如 -123 → "-123"，12 → "+012"
def split_to_2d(arr, cols=1):
    return [arr[i:i + cols] for i in range(0, len(arr), cols)]
def get_vertices(rect):
    x, y, w, h = rect
    # 计算四个顶点坐标
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_right = (x + w, y + h)
    bottom_left = (x, y + h)
    return [top_left, top_right, bottom_right, bottom_left]
def find_max(arr):
    max_size=0
    for s in range (len(arr)):
        if arr[s][2]*arr[s][3] > max_size:
            max_blob=arr[s]
            max_size = arr[s][2]*arr[s][3]
    return max_blob

 # 定义一个函数来判断两条线段是否平行
def are_segments_parallel(theta1, theta2, tolerance=30):
    # 计算角度差
    angle_difference = abs(theta1 - theta2)
    if(angle_difference>180):angle_difference=angle_difference-180
    # 检查角度差是否为0度或180度（考虑浮点数精度）
    return math.isclose(angle_difference, 0, abs_tol=tolerance) or math.isclose(angle_difference, 180, abs_tol=tolerance)
def are_segments_vertical(theta1, theta2, tolerance=30):
    # 计算角度差
    angle_difference = abs(theta1 - theta2)
    if(angle_difference>180):angle_difference=angle_difference-180
    # 检查角度差是否为0度或180度（考虑浮点数精度）
    return math.isclose(angle_difference, 90, abs_tol=tolerance)
def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    def calculate_determinant(A, B):
        return A[0] * B[1] - A[1] * B[0]

    # 向量 AB 和 AC
    AB = (x2 - x1, y2 - y1)
    AC = (x3 - x1, y3 - y1)
    # 向量 CD
    CD = (x4 - x3, y4 - y3)

    # 计算叉积 determinant(AB, CD)
    det = calculate_determinant(AB, CD)

    if det == 0:
        # 直线平行或共线，没有唯一交点
        return None

    # 向量 DA
    DA = (x1 - x4, y1 - y4)

    # 计算参数 t 和 u
    t = calculate_determinant(AC, CD) / det
    u = calculate_determinant(AC, AB) / det

    # 计算交点坐标
    intersection_x = x1 + t * AB[0]
    intersection_y = y1 + t * AB[1]

    return int(intersection_x), int(intersection_y)
#def are_segments_vertical(len1, len2, tolerance=0.2):
#        # 计算角度差
#        len_bili = 1en1/1en2
#        if(angle_difference>180):angle_difference=angle_difference-180
#        # 检查角度差是否为0度或180度（考虑浮点数精度）
#        return math.isclose(angle_difference, 90, abs_tol=tolerance)
def select_rectangle_center(rects):
    """由大到小筛选几何有效候选；大干扰框失败后继续检查其他候选。"""
    candidates = [r for r in (rects or []) if len(r) >= 12 and r[2] > 0 and r[3] > 0]
    candidates.sort(key=lambda r: r[2] * r[3], reverse=True)
    for r in candidates:
        if r[2] * r[3] <= S_THRESHOLD:
            continue
        c = [(r[4 + 2*i], r[5 + 2*i]) for i in range(4)]
        edges = ((0, 1), (2, 3), (0, 3), (1, 2))
        lengths = [sqrt((c[a][0]-c[b][0])**2 + (c[a][1]-c[b][1])**2) for a, b in edges]
        if min(lengths) <= 30:
            continue
        if abs(lengths[0]-lengths[1]) >= length_threshold or abs(lengths[2]-lengths[3]) >= length_threshold:
            continue
        angles = [math.degrees(math.atan2(c[a][1]-c[b][1], c[a][0]-c[b][0])) for a, b in edges]
        if not (are_segments_parallel(angles[0], angles[1]) and
                are_segments_parallel(angles[2], angles[3]) and
                are_segments_vertical(angles[0], angles[2])):
            continue
        center = find_intersection(c[0][0], c[0][1], c[2][0], c[2][1],
                                   c[1][0], c[1][1], c[3][0], c[3][1])
        if center is not None and 0 <= center[0] < DETECT_WIDTH and 0 <= center[1] < DETECT_HEIGHT:
            return center
    return None


def process_frame(raw, control):
    detection = prepare_detection(raw, control)
    if detection is None:
        return None
    pixels = detection.to_numpy_ref()
    rects = cv_lite.rgb888_find_rectangles_with_corners(
        image_shape, pixels, canny_thresh1, canny_thresh2,
        approx_epsilon, area_min_ratio, max_angle_cos, gaussian_blur_size)
    # detection存活到库调用结束，避免numpy引用对应的图像缓冲过早释放。
    return select_rectangle_center(rects)


def send_center(center):
    if center is None:
        # 保留旧工程的失败消息；接收端目前不解析此格式，详见集成说明。
        # 不用[+999+999*]代替，以免现有控制端将哨兵值当成运动误差。
        uart.write('(x=999,y=999)')
        return
    dx = DETECT_WIDTH // 2 - center[0]
    dy = DETECT_HEIGHT // 2 - center[1]
    uart.write('[' + format_coord(dx) + format_coord(dy) + '*]')


def camera_init():
    global sensor, media_started
    sensor = Sensor()
    sensor.reset()
    sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT)
    sensor.set_pixformat(Sensor.RGB888)
    MediaManager.init()
    media_started = True
    sensor.run()


def camera_deinit():
    global sensor, media_started
    try:
        if sensor is not None:
            sensor.stop()
    finally:
        sensor = None
        if media_started:
            MediaManager.deinit()
            media_started = False


def capture_picture():
    control = AutoThreshold()
    key = DebouncedPress(RECALIBRATE_KEY.value(), ticks_ms())
    while True:
        os.exitpoint()
        if key.update(RECALIBRATE_KEY.value(), ticks_ms()):
            control.request()
        raw = sensor.snapshot()
        center = process_frame(raw, control)
        send_center(center)
        del raw


def main():
    os.exitpoint(os.EXITPOINT_ENABLE)
    try:
        camera_init()
        capture_picture()
    except KeyboardInterrupt:
        print('user stop')
    except Exception as e:
        print('vision stopped:', e)
        raise
    finally:
        camera_deinit()


if __name__ == '__main__':
    main()
