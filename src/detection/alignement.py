import cv2
import numpy as np

class FaceAligner:
    def __init__(self):
        # Load pre-trained Haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def detect_largest_face(self, gray):
        # Relaxed parameters for better tolerance (glasses, lighting)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        if len(faces) == 0:
            return None
        # Return the largest face by area
        return max(faces, key=lambda f: f[2] * f[3])

    def detect_face_only(self, image):
        """Quickly detect face coordinates without alignment."""
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detect_largest_face(gray)

    def get_eye_centers(self, face_gray):
        h, w = face_gray.shape[:2]
        # Search only in the upper half of the face
        roi_gray = face_gray[0:int(h/2), :]
        
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        
        if len(eyes) < 2:
            # Return geometric defaults if eyes aren't found
            return [(int(w/3), int(h/3)), (int(2*w/3), int(h/3))]
        
        # Pick the two largest detections
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        
        centers = []
        for (ex, ey, ew, eh) in eyes:
            centers.append((ex + ew // 2, ey + eh // 2))
        
        # Sort by X to ensure [left_eye, right_eye]
        centers = sorted(centers, key=lambda c: c[0])
        return centers

    def align(self, image):
        if image is None:
            return None, None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_coords = self.detect_largest_face(gray)
        
        if face_coords is None:
            return None, None
            
        x, y, w, h = face_coords
        face_gray = gray[y:y+h, x:x+w]
        
        eye_centers_initial = self.get_eye_centers(face_gray)
        
        if eye_centers_initial is not None:
            # Calculate angle to align eyes horizontally
            left_eye, right_eye = eye_centers_initial
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotation center is midpoint between eyes (forced to native Python int)
            center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
            
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Align face
            aligned_face = cv2.warpAffine(face_gray, M, (w, h), flags=cv2.INTER_CUBIC)
        else:
            # If eyes not found, just use the face region as is
            aligned_face = face_gray
            
        # Normalize: Resize to 128x128 and histogram equalization
        final_face = cv2.resize(aligned_face, (128, 128), interpolation=cv2.INTER_AREA)
        final_face = cv2.equalizeHist(final_face)
        
        # Redetect eyes on the normalized image for more precision in feature extraction
        eye_centers_final = self.get_eye_centers(final_face)
        
        return final_face, eye_centers_final
