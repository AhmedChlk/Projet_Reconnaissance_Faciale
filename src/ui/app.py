import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from typing import Optional, Tuple, List
import sys
import time

# Suppress warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

from src.detection.alignement import FaceAligner
from src.contour.snake import ActiveContourSnake
from src.features.extracteur import FeatureExtractor
from src.features.dataset_manager import DatasetManager
from src.features.identifier import FaceIdentifier

class ReconnaissanceApp:
    def __init__(self):
        # Core Tools
        self.aligner = FaceAligner()
        self.snake = ActiveContourSnake()
        self.extractor = FeatureExtractor()
        self.manager = DatasetManager()
        self.identifier = FaceIdentifier(threshold=0.45) # Tolerant for various angles

        # Tkinter UI Setup
        self.root = tk.Tk()
        self.root.title("BIOMETRIC ACCESS CONTROL")
        self.root.geometry("1000x700") 
        self.root.configure(bg="#0f0f0f")
        
        # Header
        self.header = tk.Frame(self.root, bg="#000000", height=70)
        self.header.pack(fill="x")
        self.header_label = tk.Label(self.header, text="SYSTEM READY", font=("Helvetica", 22, "bold"), bg="#000000", fg="#ffffff")
        self.header_label.pack(pady=15)
        
        # Main Layout
        self.main_frame = tk.Frame(self.root, bg="#0f0f0f")
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=10)

        # Left Column: Canvas
        self.left_col = tk.Frame(self.main_frame, bg="#0f0f0f")
        self.left_col.pack(side="left", expand=True, fill="both")
        
        self.canvas = tk.Canvas(self.left_col, width=400, height=450, bg="#0f0f0f", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.door_bg = self.canvas.create_rectangle(100, 50, 300, 380, fill="#1a1a1a", outline="#333333")
        self.door = self.canvas.create_polygon(100, 50, 300, 50, 300, 380, 100, 380, fill="#4a4a4a", outline="#666666", width=2)
        self.knob = self.canvas.create_oval(270, 210, 285, 225, fill="#c0c0c0")
        self.light_bulb = self.canvas.create_oval(190, 10, 215, 30, fill="#333333")
        
        # Right Column: Controls
        self.right_col = tk.Frame(self.main_frame, bg="#0f0f0f", width=300)
        self.right_col.pack(side="right", fill="y", padx=30)
        
        btn_config = {"font": ("Helvetica", 11, "bold"), "width": 26, "height": 2, "bd": 0, "cursor": "hand2"}
        
        tk.Label(self.right_col, text="CONTROL PANEL", font=("Helvetica", 14, "bold"), bg="#0f0f0f", fg="#555555").pack(pady=20)
        
        tk.Button(self.right_col, text="START IDENTIFICATION (I)", bg="#2980b9", fg="white", command=self.trigger_identify, **btn_config).pack(pady=8)
        tk.Button(self.right_col, text="NEW ENROLLMENT (E)", bg="#27ae60", fg="white", command=self.start_enrollment, **btn_config).pack(pady=8)
        tk.Button(self.right_col, text="DATABASE STATUS", bg="#8e44ad", fg="white", command=self.show_info, **btn_config).pack(pady=8)
        tk.Button(self.right_col, text="RESET SYSTEM", bg="#e67e22", fg="white", command=self.reset_dataset, **btn_config).pack(pady=8)
        tk.Button(self.right_col, text="EXIT (Q)", bg="#c0392b", fg="white", command=self.quit, **btn_config).pack(side="bottom", pady=40)

        # Bottom Candidates
        self.candidate_frame = tk.Frame(self.root, bg="#0a0a0a", height=100)
        self.candidate_frame.pack(side="bottom", fill="x")
        self.candidate_labels = []
        for _ in range(4):
            lbl = tk.Label(self.candidate_frame, text="", font=("Helvetica", 10), bg="#0a0a0a", fg="#999999", width=20, height=3)
            lbl.pack(side="left", padx=10, expand=True)
            self.candidate_labels.append(lbl)

        # State
        self.cap = self._init_camera()
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "No camera detected!")
            sys.exit(1)
        
        self.enrolling = False
        self.enrolling_name = ""
        self.enroll_samples = []
        self.last_sample_time = 0
        self.last_top_3 = []
        self.current_vector = None
        
        # Stability
        self.last_face_coords = None
        self.face_lost_counter = 0
        self.SMOOTHING_LIMIT = 15

    def _init_camera(self):
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
        return None

    def trigger_identify(self):
        if self.current_vector is not None: self.identify(self.current_vector)
        else: self.header_label.config(text="SCAN ERROR: NO FACE", fg="#e67e22")

    def show_info(self):
        count = len(self.manager.load_dataset())
        messagebox.showinfo("System Info", f"Total signatures: {count}\nOptimized for Distance & Profile Views.")

    def reset_dataset(self):
        if messagebox.askyesno("Confirm", "Wipe all signatures?"):
            if os.path.exists(self.manager.csv_path): os.remove(self.manager.csv_path)
            self.manager._cache = None
            messagebox.showinfo("Success", "System Reset.")

    def process_frame(self, frame):
        aligned, eye_centers, face_coords = self.aligner.align(frame)
        if face_coords is not None:
            self.last_face_coords = face_coords; self.face_lost_counter = 0
        else:
            self.face_lost_counter += 1
            if self.face_lost_counter < self.SMOOTHING_LIMIT: face_coords = self.last_face_coords

        if aligned is None: return None, None, None, face_coords
        init_c = self.snake.initialize_circle((64, 64), 52, num_points=60)
        final_c = self.snake.evolve(aligned, init_c, iterations=20)
        vector = self.extractor.extract_30d_vector(eye_centers, final_c)
        return vector, aligned, eye_centers, face_coords

    def draw_overlay(self, frame, name, dist, match, top_3, face_coords):
        color = (113, 204, 46) if match else (60, 76, 231) if name else (255, 255, 255)
        if face_coords is not None:
            fx, fy, fw, fh = face_coords
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
            cv2.putText(frame, "TRACKING" if not name else "ANALYZING", (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(frame, "BIOMETRIC ACCESS CONTROL UNIT", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if name:
            cv2.rectangle(frame, (20, h-80), (w-20, h-20), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, h-80), (w-20, h-20), color, 1)
            status = f"WELCOME {name.upper()}" if match else "ACCESS DENIED"
            cv2.putText(frame, status, (40, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def update(self):
        ret, frame = self.cap.read()
        if not ret: self.root.after(10, self.update); return
        vector, aligned, _, face_coords = self.process_frame(frame)
        self.current_vector = vector; display_frame = frame.copy(); h, w = display_frame.shape[:2]

        if self.enrolling:
            now = time.time()
            # PROFILE FIX: Use face_coords for presence check, but only save if Snake (vector) is valid
            if face_coords is not None and (now - self.last_sample_time) > 0.5:
                if vector is not None:
                    self.enroll_samples.append(vector); self.last_sample_time = now
            
            steps = ["LOOK AT CAMERA", "TURN HEAD LEFT", "TURN HEAD RIGHT"]
            idx = min(len(self.enroll_samples) // 8, 2)
            cv2.rectangle(display_frame, (50, h-110), (w-50, h-20), (0,0,0), -1)
            cv2.putText(display_frame, f"TASK: {steps[idx]}", (w//2-180, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            prog = int((len(self.enroll_samples)/24)*(w-140))
            cv2.rectangle(display_frame, (70, h-50), (w-70, h-40), (40, 40, 40), -1)
            cv2.rectangle(display_frame, (70, h-50), (70+prog, h-40), (46, 204, 113)[::-1], -1)
            if len(self.enroll_samples) >= 24: self.finalize_enrollment()
            elif face_coords is None and self.face_lost_counter > self.SMOOTHING_LIMIT:
                cv2.putText(display_frame, "FACE LOST!", (w//2-80, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if self.last_top_3:
            best_n, best_d, _ = self.last_top_3[0]
            match = best_d <= self.identifier.threshold
            self.draw_overlay(display_frame, best_n, best_d, match, self.last_top_3, face_coords)
            for i, label in enumerate(self.candidate_labels):
                if i < len(self.last_top_3):
                    n, d, c = self.last_top_3[i]
                    color = "#2ecc71" if d <= self.identifier.threshold else "#e74c3c"
                    label.config(text=f"{n}\n{c:.1f}%", fg=color)
                else: label.config(text="")
        else: self.draw_overlay(display_frame, "", 0.0, False, [], face_coords)

        cv2.imshow("BIOMETRIC ACCESS", display_frame)
        if aligned is not None: cv2.imshow("ROI SCAN", aligned)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): self.quit(); return
        elif key == ord('i'): self.trigger_identify()
        elif key == ord('e') and not self.enrolling: self.start_enrollment()
        self.root.update(); self.root.after(5, self.update)

    def identify(self, vector):
        if vector is None: 
            self.header_label.config(text="ERROR: SCAN IMPOSSIBLE", fg="#e67e22")
            self.update_door(False); return
        dataset = self.manager.load_dataset()
        if not dataset: self.header_label.config(text="NO SIGNATURES FOUND", fg="#f1c40f"); return
        best_name, best_dist, match, top_3 = self.identifier.compare(vector, dataset)
        self.last_top_3 = top_3; self.update_door(match)
        if not match: self.header_label.config(text="ACCESS DENIED - UNKNOWN", fg="#e74c3c")
        else: self.header_label.config(text=f"ACCESS GRANTED: {best_name.upper()}", fg="#2ecc71")

    def start_enrollment(self):
        name = simpledialog.askstring("ENROLLMENT", "Enter Full Name:")
        if not name: return
        if self.manager.name_exists(name): messagebox.showerror("Error", "Name already taken!"); return
        self.enrolling_name = name; self.enrolling = True; self.enroll_samples = []; self.last_sample_time = time.time()
        self.header_label.config(text=f"ENROLLING: {name.upper()}", fg="#f1c40f")

    def finalize_enrollment(self):
        self.enrolling = False
        for vec in self.enroll_samples: self.manager.save_entry(self.enrolling_name, vec)
        messagebox.showinfo("Success", f"Profile created for {self.enrolling_name}")
        self.enroll_samples = []; self.header_label.config(text="SYSTEM READY", fg="white")

    def update_door(self, authorized):
        color = "#2ecc71" if authorized else "#e74c3c"
        self.canvas.itemconfig(self.light_bulb, fill=color)
        self.canvas.itemconfig(self.door, fill="#27ae60" if authorized else "#c0392b")
        if authorized: self.canvas.coords(self.door, 100, 50, 120, 70, 120, 360, 100, 380)
        else: self.canvas.coords(self.door, 100, 50, 300, 50, 300, 380, 100, 380)

    def quit(self):
        self.cap.release(); cv2.destroyAllWindows(); self.root.destroy()

    def run(self):
        self.update(); self.root.mainloop()

if __name__ == "__main__":
    app = ReconnaissanceApp(); app.run()
