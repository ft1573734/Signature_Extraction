import cv2
import numpy as np
from PIL import Image

def extract_signature(input_path, output_path):
    # ��ȡͼ��
    img = cv2.imread(input_path)
    original = img.copy()
    
    # ת��Ϊ�Ҷ�ͼ����˹ģ��
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # ����Ӧ��ֵ����
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # ��̬ѧ����ȥ�����
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # ��������
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # �����հ�����
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # ���������ҵ�������
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # �Ż������Ե
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # ת��ΪPIL Image����͸����
    pil_img = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)
    
    # ���Alphaͨ��
    pil_img.putalpha(mask_pil)
    
    # �ü�����Ч����
    bbox = mask_pil.getbbox()
    if bbox:
        pil_img = pil_img.crop(bbox)
    
    # ������
    pil_img.save(output_path, "PNG")

if __name__ == "__main__":
    input_image = "input_signature.jpg"  # �����ļ�·��
    output_image = "signature_output.png"  # ����ļ�·��
    extract_signature(input_image, output_image)