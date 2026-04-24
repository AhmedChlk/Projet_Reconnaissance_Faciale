import cv2
import numpy as np
from typing import Tuple, Optional, List

class FaceAligner:
    def __init__(self):
        # Load multiple cascades for maximum robustness
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def detect_largest_face(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # Optimization: detect on a fixed width for stability and speed
        height, width = gray.shape
        target_w = 640
        scale_factor = width / target_w
        gray_small = cv2.resize(gray, (target_w, int(height / scale_factor)))

        # Dynamic CLAHE for low-light/distance robustness
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        gray_enhanced = clahe.apply(gray_small)
        
        configs = [
            (self.face_alt_cascade, 1.05, 3), # Alt2 is best for distance
            (self.face_cascade, 1.1, 3),     # Default
            (self.profile_cascade, 1.05, 2),  # Side profiles
        ]
        
        best_face = None
        max_area = 0
        
        for cascade, scale, min_n in configs:
            # Very small minSize (15, 15) to detect faces very far away
            faces = cascade.detectMultiScale(gray_enhanced, scaleFactor=scale, minNeighbors=min_n, minSize=(15, 15))
            for (fx, fy, fw, fh) in faces:
                if fw * fh > max_area:
                    max_area = fw * fh
                    best_face = (fx, fy, fw, fh)
            if best_face and max_area > (target_w * 0.15)**2: break # Found decent face
            
        if best_face is None:
            flipped = cv2.flip(gray_enhanced, 1)
            faces_flipped = self.profile_cascade.detectMultiScale(flipped, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15))
            if len(faces_flipped) > 0:
                fx, fy, fw, fh = max(faces_flipped, key=lambda f: f[2] * f[3])
                fx = target_w - fx - fw
                best_face = (fx, fy, fw, fh)
            
        if best_face:
            fx, fy, fw, fh = best_face
            return (int(fx * scale_factor), int(fy * scale_factor), int(fw * scale_factor), int(fh * scale_factor))
        return None

    def get_eye_centers(self, face_gray: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        h, w = face_gray.shape
        roi_y1, roi_y2 = int(h * 0.15), int(h * 0.6)
        upper_half = face_gray[roi_y1:roi_y2, :]
        for min_n in [5, 3, 2]:
            eyes = self.eye_cascade.detectMultiScale(upper_half, scaleFactor=1.05, minNeighbors=min_n)
            if len(eyes) >= 2: break
        if len(eyes) < 2: return None
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        centers = sorted([(ex + ew // 2, ey + eh // 2 + roi_y1) for (ex, ey, ew, eh) in eyes], key=lambda c: c[0])
        return centers

    def align(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[int, int]]], Optional[Tuple[int, int, int, int]]]:
        if image is None: return None, None, None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_coords = self.detect_largest_face(gray)
        if face_coords is None: return None, None, None
            
        x, y, w, h = face_coords
        face_gray = gray[y:y+h, x:x+w]
        eye_centers = self.get_eye_centers(face_gray)
        
        if eye_centers and len(eye_centers) >= 2:
            left_eye, right_eye = eye_centers
            dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_face = cv2.warpAffine(face_gray, M, (w, h), flags=cv2.INTER_CUBIC)
        else:
            aligned_face = face_gray
            
        final_face = cv2.resize(aligned_face, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        final_face = clahe.apply(final_face)
        return final_face, self.get_eye_centers(final_face), face_coords
