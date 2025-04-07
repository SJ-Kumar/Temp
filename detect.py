import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from imgocr import OcrEngine

ocr = OcrEngine(lang='eng')

# Step 1: Extract field regions (bounding boxes) from the template image
def get_template_fields_via_ocr(template_img):
    ocr_results = ocr.image_to_string(template_img, with_boxes=True)
    field_boxes = {}
    for i, field in enumerate(ocr_results):
        x1, y1, x2, y2 = field['box']
        label = field['word'].strip().lower() if field['word'].strip() else f"field_{i}"
        field_boxes[label] = (x1, y1, x2, y2)
    return field_boxes

# Step 2: Structural tampering detection using SSIM
def detect_structural_tampering(template_img, test_img, field_boxes):
    output = test_img.copy()
    tampered = []
    for field, (x1, y1, x2, y2) in field_boxes.items():
        try:
            t_crop = cv2.cvtColor(template_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            test_crop = cv2.cvtColor(test_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

            t_crop = cv2.resize(t_crop, (100, 40))
            test_crop = cv2.resize(test_crop, (100, 40))

            score = ssim(t_crop, test_crop)
            if score < 0.7:
                tampered.append((field, (x1, y1, x2, y2), score))
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(output, f"Tampered: {field}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        except:
            continue
    return output, tampered

# Step 3: Font style inconsistency detection using SSIM in OCR regions
def detect_font_issues(template_img, test_img, field_boxes):
    output = test_img.copy()
    mismatched = []
    for field, (x1, y1, x2, y2) in field_boxes.items():
        try:
            t_crop = cv2.cvtColor(template_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            test_crop = cv2.cvtColor(test_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

            t_crop = cv2.resize(t_crop, (100, 40))
            test_crop = cv2.resize(test_crop, (100, 40))

            score = ssim(t_crop, test_crop)
            if score < 0.7:
                mismatched.append((field, (x1, y1, x2, y2), score))
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(output, f"Font Mismatch: {field}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        except:
            continue
    return output, mismatched

# Step 4: Run the detection pipeline
def detect_id_card_tampering(template_path, test_path):
    template = cv2.imread(template_path)
    test = cv2.imread(test_path)

    if template.shape != test.shape:
        test = cv2.resize(test, (template.shape[1], template.shape[0]))

    field_boxes = get_template_fields_via_ocr(template)

    tamper_result, tampered_fields = detect_structural_tampering(template, test, field_boxes)
    font_result, font_issues = detect_font_issues(template, test, field_boxes)

    final_result = cv2.addWeighted(tamper_result, 0.5, font_result, 0.5, 0)

    print("\n--- Tampered Fields ---")
    for field, box, score in tampered_fields:
        print(f"{field} | SSIM: {score:.3f} | Box: {box}")

    print("\n--- Font Inconsistencies ---")
    for field, box, score in font_issues:
        print(f"{field} | SSIM: {score:.3f} | Box: {box}")

    cv2.imshow("Tampering Detection Result", final_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_id_card_tampering("template.jpg", "test.jpg")