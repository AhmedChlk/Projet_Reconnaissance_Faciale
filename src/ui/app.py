import os
import threading
from collections import Counter
# Supprime les warnings Qt liés à OpenCV et les logs verbeux
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

import cv2
import numpy as np
import tkinter as tk
import sys
from tkinter import simpledialog
from PIL import Image, ImageTk
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
        self.identifier = FaceIdentifier(threshold=0.65)

        # UI (Tkinter Window)
        self.root = tk.Tk()
        self.root.title("Système de Reconnaissance Faciale")
        self.root.geometry("950x600")
        
        # Main Layout
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Video Section (Left)
        self.video_container = tk.Frame(self.main_container)
        self.video_container.pack(side="left", fill="both", expand=True)
        
        self.video_label = tk.Label(self.video_container, bg="black")
        self.video_label.pack(fill="both", expand=True)
        
        # Door Section (Right)
        self.door_section = tk.Frame(self.main_container, width=250)
        self.door_section.pack(side="right", fill="y", padx=10)
        
        # 3D Door Canvas
        self.canvas = tk.Canvas(self.door_section, width=200, height=350, bg="#333", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        # Draw Wall
        self.canvas.create_rectangle(10, 10, 190, 340, fill="#555", outline="#777")
        
        # Door Coordinates
        self.door_closed = [40, 50, 160, 50, 160, 300, 40, 300]
        self.door_open = [40, 50, 120, 80, 120, 270, 40, 300]
        
        # Create Door Object
        self.door_id = self.canvas.create_polygon(self.door_closed, fill="#8B4513", outline="#5D2E0A", width=2)
        
        # Create Handle
        self.handle_id = self.canvas.create_oval(140, 170, 150, 180, fill="gold", outline="black")
        
        # Create Light (Voyant)
        self.voyant_id = self.canvas.create_oval(85, 20, 115, 50, fill="black", outline="#222")
        
        self.status_label = tk.Label(self.door_section, text="Système Prêt", font=("Arial", 12, "bold"))
        self.status_label.pack(pady=10)
        
        self.info_label = tk.Label(self.door_section, text="Raccourcis :\n'I' : Identifier\n'E' : Enregistrer\n'Q' : Quitter", justify="left")
        self.info_label.pack(side="bottom", pady=20)

        # Key bindings
        self.root.bind('<i>', lambda e: self.trigger_identify())
        self.root.bind('<e>', lambda e: self.start_enrollment())
        self.root.bind('<q>', lambda e: self.quit_app())

        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Erreur critique : Aucune caméra détectée.")
            sys.exit(1)
        
        # State
        self.enrolling = False
        self.enroll_samples = []
        self.last_result = ("Aucun", 0.0, False, 0.0, []) # Name, Dist, Match, Conf, Top3
        self.frame_count = 0
        self.recent_predictions = []
        self.smoothed_box = None
        self.current_face_aligned = None

    def process_frame(self, frame):
        aligned, eye_centers = self.aligner.align(frame)
        if aligned is None or eye_centers is None:
            return None, None
        initial_contour = self.snake.initialize_circle((64, 64), 50, num_points=60)
        final_contour = self.snake.evolve(aligned, initial_contour, iterations=20)
        vector = self.extractor.extract_30d_vector(eye_centers, final_contour)
        return vector, aligned

    def trigger_identify(self):
        ret, frame = self.cap.read()
        if not ret: return
        try:
            self.status_label.config(text="Analyse...")
            self.root.update()
            vector, aligned = self.process_frame(frame)
            self.current_face_aligned = aligned
            self.identify(vector)
        except Exception as e:
            print(f"Erreur d'identification : {e}")
            self.last_result = ("Erreur", 0.0, False, 0.0, [])

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            err_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_img, "Webcam introuvable", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.show_frame(err_img)
            self.root.after(10, self.update)
            return

        display_frame = frame.copy()
        box = self.aligner.detect_face_only(frame)
        if box is not None:
            if self.smoothed_box is None:
                self.smoothed_box = box
            else:
                alpha = 0.2
                self.smoothed_box = tuple(int(alpha * n + (1 - alpha) * o) for n, o in zip(box, self.smoothed_box))
            x, y, w, h = self.smoothed_box
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            self.smoothed_box = None

        if self.enrolling:
            self.frame_count += 1
            if self.frame_count % 5 == 0:
                try:
                    vector, aligned = self.process_frame(frame)
                    self.current_face_aligned = aligned
                    if vector is not None:
                        self.enroll_samples.append(vector)
                        if len(self.enroll_samples) >= 20:
                            self.finalize_enrollment()
                except Exception as e:
                    print(f"Erreur d'enrôlement : {e}")
            cv2.putText(display_frame, f"Capture {len(self.enroll_samples)}/20", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Overlay results
        name, dist, match, conf, top_3 = self.last_result
        h, w = display_frame.shape[:2]
        
        # Transparent banner
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        color = (0, 255, 0) if match else (255, 255, 255)
        txt = f"Nom: {name} | Dist: {dist:.2f} | Confiance: {conf:.1f}%"
        cv2.putText(display_frame, txt, (15, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display Top 3
        if top_3:
            top_txt = " | ".join([f"{i+1}. {n} ({d:.2f})" for i, (n, d) in enumerate(top_3)])
            cv2.putText(display_frame, top_txt, (15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if self.current_face_aligned is not None:
            thumb = cv2.resize(self.current_face_aligned, (64, 64))
            if len(thumb.shape) == 2: thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            display_frame[10:74, w-74:w-10] = thumb
            cv2.rectangle(display_frame, (w-75, 9), (w-9, 75), (255, 255, 255), 1)

        self.show_frame(display_frame)
        self.root.after(10, self.update)

    def show_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk

    def identify(self, vector):
        if vector is None:
            self.recent_predictions.append(("Inconnu", 0.0, 0.0, []))
        else:
            dataset = self.manager.load_dataset()
            name, dist, match, confidence, top_3 = self.identifier.compare(vector, dataset)
            if match:
                self.recent_predictions.append((name, dist, confidence, top_3))
            else:
                self.recent_predictions.append(("Inconnu", dist, confidence, top_3))
        
        if len(self.recent_predictions) > 5:
            self.recent_predictions.pop(0)
            
        if self.recent_predictions:
            names = [p[0] for p in self.recent_predictions]
            counts = Counter(names)
            final_name, count = counts.most_common(1)[0]
            last_stats = next(p for p in reversed(self.recent_predictions) if p[0] == final_name)
            _, last_dist, last_conf, last_top3 = last_stats
            
            if final_name != "Inconnu":
                self.last_result = (final_name, last_dist, True, last_conf, last_top3)
                self.update_door(True)
            else:
                self.last_result = ("Inconnu", last_dist, False, last_conf, last_top3)
                self.update_door(False)

    def start_enrollment(self):
        self.enrolling = True
        self.enroll_samples = []
        self.frame_count = 0
        self.status_label.config(text="Cadrage en cours...")

    def finalize_enrollment(self):
        self.enrolling = False
        samples = self.enroll_samples.copy()
        self.enroll_samples = []
        name = simpledialog.askstring("Enregistrement", "Entrez le nom de la personne :", parent=self.root)
        if name:
            def save_thread():
                avg_vector = np.mean(samples, axis=0)
                self.manager.save_entry(name, avg_vector)
                print(f"✅ Dataset mis à jour : {name}")
                self.root.after(0, lambda: self.status_label.config(text=f"Enregistré : {name}"))
            threading.Thread(target=save_thread, daemon=True).start()
        else:
            self.status_label.config(text="Annulé")

    def update_door(self, authorized):
        if authorized:
            self.canvas.coords(self.door_id, *self.door_open)
            self.canvas.itemconfig(self.handle_id, state="hidden")
            self.canvas.itemconfig(self.voyant_id, fill="lime")
            self.status_label.config(text="ACCES AUTORISE", fg="green")
        else:
            self.canvas.coords(self.door_id, *self.door_closed)
            self.canvas.itemconfig(self.handle_id, state="normal")
            self.canvas.itemconfig(self.voyant_id, fill="red")
            self.status_label.config(text="ACCES REFUSE", fg="red")

    def quit_app(self):
        self.cap.release()
        self.root.destroy()
        sys.exit(0)

    def run(self):
        self.update()
        self.root.mainloop()

if __name__ == "__main__":
    app = ReconnaissanceApp()
    app.run()
