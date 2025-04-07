import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from imgocr import OcrEngine

ocr = OcrEngine(lang='eng')

# Step 1: Align the test image to the template
def align_images(template_img, test_img):
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_template, None)
    kp2, des2 = orb.detectAndCompute(gray_test, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    aligned_test = cv2.warpPerspective(test_img, matrix, (template_img.shape[1], template_img.shape[0]))

    return aligned_test

# Step 2: Tampering detection using SSIM
def detect_visual_tampering(template_img, aligned_img):
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray_template, gray_aligned, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(template_img)
    for c in contours:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return mask, contours

# Step 3: Font style consistency check using image patches
def detect_font_mismatch(template_img, aligned_img):
    template_ocr = ocr.image_to_string(template_img, with_boxes=True)
    aligned_ocr = ocr.image_to_string(aligned_img, with_boxes=True)

    font_mismatch_boxes = []

    for t, a in zip(template_ocr, aligned_ocr):
        x1, y1, x2, y2 = t['box']
        t_crop = cv2.cvtColor(template_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        a_crop = cv2.cvtColor(aligned_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        # Resize both to same shape for fair comparison
        try:
            t_crop = cv2.resize(t_crop, (50, 50))
            a_crop = cv2.resize(a_crop, (50, 50))

            font_diff_score = ssim(t_crop, a_crop)
            if font_diff_score < 0.7:
                font_mismatch_boxes.append(((x1, y1, x2, y2), font_diff_score))
        except:
            continue

    return font_mismatch_boxes

# Step 4: Visualize all anomalies
def visualize_results(image, tampered_contours, font_mismatches):
    output = image.copy()

    # Draw SSIM tampering regions
    for c in tampered_contours:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(output, "Tampered", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw font mismatch regions
    for box, score in font_mismatches:
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(output, f"Font Mismatch ({score:.2f})", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return output

# === Main Execution ===
template = cv2.imread("template.jpg")
test = cv2.imread("test.jpg")

aligned = align_images(template, test)
tamper_mask, tamper_contours = detect_visual_tampering(template, aligned)
font_mismatches = detect_font_mismatch(template, aligned)
final_output = visualize_results(aligned, tamper_contours, font_mismatches)

cv2.imshow("Tampering Detected", final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()