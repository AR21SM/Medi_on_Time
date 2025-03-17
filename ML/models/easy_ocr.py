import easyocr
import cv2
import numpy as np

def get_ocr_text(image_path):

    # Load Image
    image = cv2.imread(image_path)

    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Adaptive Thresholding
    processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Save and Reload Processed Image for OCR
    cv2.imwrite("ML/models/testimages/processed_image.jpg", processed_img)

    # Initialize EasyOCR Reader (English)
    reader = easyocr.Reader(['en'], gpu=False)

    # Perform OCR
    result = reader.readtext("ML/models/testimages/processed_image.jpg", detail=0, paragraph=True)

    return "\n".join(result)  

image_path = "ML/models/testimages/a4.png"
res=get_ocr_text(image_path)
print(res)
