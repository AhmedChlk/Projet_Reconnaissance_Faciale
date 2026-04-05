import os
# Supprime les warnings Qt liés à OpenCV et les logs verbeux
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

import cv2
import numpy as np
import tkinter as tk
import sys
from tkinter import simpledialog
from src.detection.alignement import FaceAligner
from src.contour.snake import ActiveContourSnake
from src.features.extracteur import FeatureExtractor
from src.features.dataset_manager import DatasetManager
from src.features.identifier import FaceIdentifier

class ReconnaissanceApp:
    def __init__(self):
        # Tools
        self.aligner = FaceAligner()
        self.snake = ActiveContourSnake()
        self.extractor = FeatureExtractor()
        self.manager = DatasetManager()
        self.identifier = FaceIdentifier(threshold=0.35)

        # UI (Tkinter Window)
        self.root = tk.Tk()
        self.root.title("Porte de Sécurité")
        self.root.geometry("200x300")
        
        self.door_frame = tk.Frame(self.root, bg="gray", width=180, height=250)
        self.door_frame.pack(pady=10)
        
        self.light = tk.Label(self.door_frame, bg="black", width=5, height=2)
        self.light.place(relx=0.5, rely=0.2, anchor="center")
        
        self.status_label = tk.Label(self.root, text="En attente...")
        self.status_label.pack()

        # Webcam Initialization with check
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Erreur critique : Aucune caméra détectée sur l'index 0.")
            print("Vérifiez vos branchements, vos permissions ou si une autre application utilise la caméra.")
            sys.exit(1)
        
        # State
        self.enrolling = False
        self.enroll_samples = []
        self.last_result = ("Aucun", 0.0, False)

    def process_frame(self, frame):
        """Align face and extract features if possible."""
        aligned, eye_centers = self.aligner.align(frame)
        if aligned is None or eye_centers is None:
            return None, None
            
        # For Snake, we need initial circle (center of face area is middle of 128x128)
        initial_contour = self.snake.initialize_circle((64, 64), 50, num_points=60)
        final_contour = self.snake.evolve(aligned, initial_contour, iterations=20)
        
        vector = self.extractor.extract_30d_vector(eye_centers, final_contour)
        return vector, aligned

    def update(self):
        ret, frame = self.cap.read()
        
        if not ret:
            display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(display_frame, "Erreur: Webcam introuvable", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            vector, aligned = None, None
        else:
            display_frame = frame.copy()
            vector, aligned = self.process_frame(frame)

            if vector is not None:
                # Draw something on display_frame to show detection
                cv2.putText(display_frame, "Visage Detecte", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show the aligned face in a small window overlay or separate window
                cv2.imshow("Normalized Face", aligned)

        # Overlay results
        name, dist, match = self.last_result
        color = (0, 255, 0) if match else (0, 0, 255)
        txt = f"{name} (Dist: {dist:.2f})"
        cv2.putText(display_frame, txt, (10, display_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.enrolling:
            cv2.putText(display_frame, f"Enregistrement: {len(self.enroll_samples)}/20", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if vector is not None:
                self.enroll_samples.append(vector)
                if len(self.enroll_samples) >= 20:
                    self.finalize_enrollment()

        cv2.imshow("Webcam - 'I': Id, 'E': Enroll, 'Q': Quit", display_frame)

        # Keyboard Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.quit()
            return # Stop the loop
        elif key == ord('i'):
            self.identify(vector)
        elif key == ord('e'):
            if not self.enrolling:
                self.start_enrollment()

        self.root.update()
        self.root.after(10, self.update)

    def identify(self, vector):
        if vector is None:
            self.last_result = ("Pas de visage", 0.0, False)
            self.update_door(False)
            return

        dataset = self.manager.load_dataset()
        name, dist, match = self.identifier.compare(vector, dataset)
        
        if name is None:
            self.last_result = ("Inconnu (Dataset vide)", 0.0, False)
        else:
            self.last_result = (name, dist, match)
            
        self.update_door(match)

    def start_enrollment(self):
        self.enrolling = True
        self.enroll_samples = []
        self.status_label.config(text="Capture en cours (bougez légèrement)...")

    def finalize_enrollment(self):
        self.enrolling = False
        avg_vector = np.mean(self.enroll_samples, axis=0)
        name = simpledialog.askstring("Input", "Entrez le nom de la personne :")
        if name:
            self.manager.save_entry(name, avg_vector)
            self.status_label.config(text=f"Enregistré : {name}")
        else:
            self.status_label.config(text="Enregistrement annulé.")
        self.enroll_samples = []

    def update_door(self, authorized):
        if authorized:
            self.light.config(bg="lime")
            self.door_frame.config(bg="green")
            self.status_label.config(text="ACCES AUTORISE", fg="green")
        else:
            self.light.config(bg="red")
            self.door_frame.config(bg="darkred")
            self.status_label.config(text="ACCES REFUSE", fg="red")

    def quit(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self):
        self.update()
        self.root.mainloop()

if __name__ == "__main__":
    app = ReconnaissanceApp()
    app.run()
