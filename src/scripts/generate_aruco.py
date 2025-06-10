import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- 参数设置 ---
ARUCO_DICT_NAME = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.04  # 米
MARKER_SEPARATION = 0.01  # 米
BOARD_ROWS = 4
BOARD_COLS = 6

# --- 创建字典和棋盘板 ---
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
board = cv2.aruco.GridBoard((BOARD_COLS, BOARD_ROWS), MARKER_LENGTH, MARKER_SEPARATION, aruco_dict)

# --- 计算物理大小并转换为像素 ---
DPI = 300  # 打印分辨率，单位 dpi
cm_per_inch = 2.54
width_cm = 29
height_cm = 19

width_inch = width_cm / cm_per_inch
height_inch = height_cm / cm_per_inch
width_px = int(width_inch * DPI)
height_px = int(height_inch * DPI)

# --- 生成图像 ---
board_image = board.generateImage((width_px, height_px), marginSize=20, borderBits=1)

# --- 保存为 PNG 临时图像 ---
temp_png_path = "aruco_board_tmp.png"
cv2.imwrite(temp_png_path, board_image)

# --- 生成 PDF ---
pdf_path = "aruco_board_29x19cm.pdf"
with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(width_inch, height_inch), dpi=DPI)
    img = plt.imread(temp_png_path)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print(f"✅ 已生成 PDF 文件：{pdf_path}（尺寸精确为 29×19 cm）")
