import cv2
import numpy as np
from PIL import Image

def extract_signature(input_path, output_path):
    # 读取图像
    img = cv2.imread(input_path)
    original = img.copy()
    
    # 转换为灰度图并高斯模糊
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 形态学操作去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建空白掩码
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # 绘制所有找到的轮廓
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # 优化掩码边缘
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # 转换为PIL Image处理透明度
    pil_img = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)
    
    # 添加Alpha通道
    pil_img.putalpha(mask_pil)
    
    # 裁剪到有效区域
    bbox = mask_pil.getbbox()
    if bbox:
        pil_img = pil_img.crop(bbox)
    
    # 保存结果
    pil_img.save(output_path, "PNG")

if __name__ == "__main__":
    input_image = "input_signature.jpg"  # 输入文件路径
    output_image = "signature_output.png"  # 输出文件路径
    extract_signature(input_image, output_image)