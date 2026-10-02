"""
GUI Application for Two-Stage Ordinal Cow BCS Estimation Engine.

Provides an interactive graphical dashboard to load cow images, visualize YOLO
bounding box detections, display cropped anatomical regions, predict Body Condition
Score (BCS) using an ordinal classification model, inspect threshold probabilities,
and verify ground truth (actual) scores derived from dataset directory structures.
"""

import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Set
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw

try:
    from inference import BCSInferenceEngine
except ImportError:
    from ReNet18_ordinal_classification_model.inference import BCSInferenceEngine

# Valid BCS directory labels used for ground-truth extraction
VALID_BCS_FOLDERS: Set[str] = {"3.25", "3.5", "3.75", "4", "4.0", "4.25"}


class OrdinalBCSApp(tk.Tk):
    """
    Tkinter-based GUI dashboard for the two-stage Cow BCS estimation engine.
    """

    def __init__(self) -> None:
        """Initialize application window, internal state variables, UI styling, and background model loader."""
        super().__init__()
        self.title("Cow BCS Estimation - Ordinal Classifier Dashboard")
        self.geometry("1120x780")
        self.minsize(920, 650)
        self.configure(bg="#1e1e2e")

        # Internal state tracking
        self.engine: Optional[BCSInferenceEngine] = None
        self.current_image_path: Optional[str] = None
        self.loaded_image: Optional[Image.Image] = None
        self.tk_display_img: Optional[ImageTk.PhotoImage] = None
        self.tk_crop_img: Optional[ImageTk.PhotoImage] = None
        self.actual_bcs: Optional[str] = None

        self._setup_styles()
        self._create_widgets()
        self._init_engine_thread()

    def _setup_styles(self) -> None:
        """Configure TTK widget themes, styles, colors, and typography."""
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.style.configure("TFrame", background="#1e1e2e")
        self.style.configure("Card.TFrame", background="#252538", relief="flat")
        self.style.configure(
            "Primary.TButton",
            background="#3b82f6",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=8,
        )
        self.style.map("Primary.TButton", background=[("active", "#2563eb")])

        self.style.configure(
            "Success.TButton",
            background="#10b981",
            foreground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            borderwidth=0,
            padding=10,
        )
        self.style.map("Success.TButton", background=[("active", "#059669")])

        self.style.configure(
            "Info.TButton",
            background="#8b5cf6",
            foreground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
            padding=6,
        )
        self.style.map("Info.TButton", background=[("active", "#7c3aed")])

        self.style.configure(
            "Header.TLabel",
            background="#1e1e2e",
            foreground="#f8fafc",
            font=("Segoe UI", 16, "bold"),
        )
        self.style.configure(
            "SubHeader.TLabel",
            background="#252538",
            foreground="#94a3b8",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "ResultValue.TLabel",
            background="#252538",
            foreground="#38bdf8",
            font=("Segoe UI", 24, "bold"),
        )
        self.style.configure(
            "ActualValue.TLabel",
            background="#252538",
            foreground="#a78bfa",
            font=("Segoe UI", 13, "bold"),
        )

    def _create_widgets(self) -> None:
        """Build and layout UI components and layout panels."""
        # 1. Header Bar
        header_frame = ttk.Frame(self)
        header_frame.pack(fill="x", padx=20, pady=(15, 10))

        title_lbl = ttk.Label(
            header_frame,
            text="🐄 Two-Stage Ordinal Cow BCS Predictor",
            style="Header.TLabel",
        )
        title_lbl.pack(side="left")

        self.status_lbl = ttk.Label(
            header_frame,
            text="⏳ Initializing models...",
            background="#1e1e2e",
            foreground="#fbbf24",
            font=("Segoe UI", 10, "italic"),
        )
        self.status_lbl.pack(side="right")

        # 2. Main Body Container
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=20, pady=10)

        # Left Panel (Input Image & YOLO Detection)
        left_panel = ttk.Frame(main_container, style="Card.TFrame")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        lbl_img_title = ttk.Label(
            left_panel,
            text="Camera Feed / YOLO Detection",
            style="SubHeader.TLabel",
        )
        lbl_img_title.pack(anchor="w", padx=15, pady=(10, 5))

        self.img_canvas = tk.Canvas(left_panel, bg="#181825", highlightthickness=0)
        self.img_canvas.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Right Panel (Cropped region & Ordinal Predictions)
        right_panel = ttk.Frame(main_container, width=380, style="Card.TFrame")
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        # Action Buttons
        btn_frame = ttk.Frame(right_panel, style="Card.TFrame")
        btn_frame.pack(fill="x", padx=15, pady=15)

        self.btn_select = ttk.Button(
            btn_frame,
            text="📁 Select Cow Image",
            style="Primary.TButton",
            command=self.select_image,
        )
        self.btn_select.pack(fill="x", pady=(0, 8))

        self.btn_predict = ttk.Button(
            btn_frame,
            text="⚡ Run BCS Prediction",
            style="Success.TButton",
            state="disabled",
            command=self.run_prediction_thread,
        )
        self.btn_predict.pack(fill="x")

        # Cropped Region Preview
        crop_title = ttk.Label(
            right_panel, text="Cropped Region (Stage 2 Input)", style="SubHeader.TLabel"
        )
        crop_title.pack(anchor="w", padx=15, pady=(5, 5))

        self.crop_canvas = tk.Canvas(
            right_panel, height=130, bg="#181825", highlightthickness=0
        )
        self.crop_canvas.pack(fill="x", padx=15, pady=(0, 10))

        # Result Displays
        res_box = ttk.Frame(right_panel, style="Card.TFrame")
        res_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        res_title = ttk.Label(res_box, text="Estimated BCS Score", style="SubHeader.TLabel")
        res_title.pack(anchor="w")

        self.bcs_result_lbl = ttk.Label(res_box, text="--", style="ResultValue.TLabel")
        self.bcs_result_lbl.pack(pady=2)

        self.confidence_lbl = ttk.Label(
            res_box,
            text="Prediction Certainty: -- %",
            background="#252538",
            foreground="#e2e8f0",
            font=("Segoe UI", 10),
        )
        self.confidence_lbl.pack()

        # Actual Ground Truth BCS Frame
        actual_frame = ttk.Frame(res_box, style="Card.TFrame")
        actual_frame.pack(fill="x", pady=(8, 2))

        self.btn_show_actual = ttk.Button(
            actual_frame,
            text="👁️ Show Actual BCS",
            style="Info.TButton",
            state="disabled",
            command=self.show_actual_bcs,
        )
        self.btn_show_actual.pack(fill="x")

        self.lbl_actual_bcs = ttk.Label(
            actual_frame,
            text="",
            style="ActualValue.TLabel",
            anchor="center",
        )
        self.lbl_actual_bcs.pack(pady=(4, 0))

        # Ordinal Threshold Probabilities Breakdown
        prob_title = ttk.Label(
            res_box, text="Ordinal Threshold Probabilities P(Y > k):", style="SubHeader.TLabel"
        )
        prob_title.pack(anchor="w", pady=(8, 5))

        self.prob_text = tk.Text(
            res_box,
            height=5,
            bg="#181825",
            fg="#cbd5e1",
            relief="flat",
            font=("Consolas", 9),
            padx=8,
            pady=6,
        )
        self.prob_text.pack(fill="both", expand=True)
        self.prob_text.config(state="disabled")

    def _init_engine_thread(self) -> None:
        """Asynchronously load inference models in a background thread to prevent UI freezing."""
        def _loader() -> None:
            try:
                self.engine = BCSInferenceEngine()
                self.status_lbl.config(text="● Model Ready", foreground="#4ade80")
                if self.loaded_image is not None:
                    self.btn_predict.config(state="normal")
            except Exception as e:
                self.status_lbl.config(text="❌ Model Load Failed", foreground="#f87171")
                messagebox.showerror("Error", f"Failed to initialize models:\n{e}")

        threading.Thread(target=_loader, daemon=True).start()

    def select_image(self) -> None:
        """
        Open a file selection dialog, load the image, extract ground truth, and reset UI state.
        """
        file_path = filedialog.askopenfilename(
            title="Select Cow Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not file_path:
            return

        self.current_image_path = file_path
        self.loaded_image = Image.open(file_path).convert("RGB")
        self._display_main_image(self.loaded_image)

        # Extract and validate parent directory name for ground-truth score
        parent_folder_name = Path(file_path).parent.name.strip()
        if parent_folder_name in VALID_BCS_FOLDERS:
            # Standardize label representation
            self.actual_bcs = "4" if parent_folder_name in ["4", "4.0"] else parent_folder_name
        else:
            self.actual_bcs = "unknown"

        # Reset states
        self.crop_canvas.delete("all")
        self.bcs_result_lbl.config(text="--")
        self.confidence_lbl.config(text="Prediction Certainty: -- %")
        self.btn_show_actual.config(state="disabled")
        self.lbl_actual_bcs.config(text="")
        self._set_prob_text("")

        if self.engine is not None:
            self.btn_predict.config(state="normal")

    def show_actual_bcs(self) -> None:
        """Display the actual ground-truth BCS score extracted from the directory structure."""
        if self.actual_bcs is not None:
            self.lbl_actual_bcs.config(text=f"Actual BCS: {self.actual_bcs}")

    def _display_main_image(self, pil_img: Image.Image) -> None:
        """
        Render the primary image on the main canvas with aspect-ratio preserving scaling.

        Args:
            pil_img (Image.Image): PIL Image object to display.
        """
        canvas_w = self.img_canvas.winfo_width() or 500
        canvas_h = self.img_canvas.winfo_height() or 450

        img_copy = pil_img.copy()
        img_copy.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)

        self.tk_display_img = ImageTk.PhotoImage(img_copy)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=self.tk_display_img, anchor="center"
        )

    def _display_crop_image(self, pil_crop: Image.Image) -> None:
        """
        Render the cropped anatomical region on the preview canvas.

        Args:
            pil_crop (Image.Image): Cropped PIL Image object.
        """
        crop_w = self.crop_canvas.winfo_width() or 300
        crop_h = self.crop_canvas.winfo_height() or 130

        img_copy = pil_crop.copy()
        img_copy.thumbnail((crop_w, crop_h), Image.Resampling.LANCZOS)

        self.tk_crop_img = ImageTk.PhotoImage(img_copy)
        self.crop_canvas.delete("all")
        self.crop_canvas.create_image(
            crop_w // 2, crop_h // 2, image=self.tk_crop_img, anchor="center"
        )

    def _set_prob_text(self, text: str) -> None:
        """
        Update the content of the read-only ordinal probabilities text box.

        Args:
            text (str): Formatted probability string to display.
        """
        self.prob_text.config(state="normal")
        self.prob_text.delete("1.0", tk.END)
        self.prob_text.insert(tk.END, text)
        self.prob_text.config(state="disabled")

    def run_prediction_thread(self) -> None:
        """Execute the two-stage inference pipeline asynchronously in a background thread."""
        if not self.loaded_image or not self.engine:
            return

        self.btn_predict.config(state="disabled", text="Predicting...")
        self.btn_show_actual.config(state="disabled")
        self.lbl_actual_bcs.config(text="")
        self.status_lbl.config(text="⚙️ Processing...", foreground="#fbbf24")

        def _predict() -> None:
            try:
                if self.engine.detector is not None:
                    results = self.engine.predict_full_pipeline(self.loaded_image)
                else:
                    bcs, conf, probs = self.engine.predict_crop(self.loaded_image)
                    results = {
                        "detected": False,
                        "bbox": None,
                        "crop_image": self.loaded_image,
                        "predicted_bcs": bcs,
                        "confidence": conf,
                        "threshold_probabilities": probs,
                    }
                self.after(0, self._on_prediction_done, results)
            except Exception as e:
                self.after(0, self._on_prediction_error, str(e))

        threading.Thread(target=_predict, daemon=True).start()

    def _on_prediction_done(self, results: Dict[str, Any]) -> None:
        """
        Handle prediction results on the main UI thread and update all visuals.

        Args:
            results (Dict[str, Any]): Dictionary containing detection and classification outputs.
        """
        self.btn_predict.config(state="normal", text="⚡ Run BCS Prediction")
        self.status_lbl.config(text="● Done", foreground="#4ade80")

        # 1. Visualize bounding box on original image
        vis_img = self.loaded_image.copy()
        if results.get("detected") and results.get("bbox"):
            draw = ImageDraw.Draw(vis_img)
            draw.rectangle(results["bbox"], outline="#ef4444", width=4)
        self._display_main_image(vis_img)

        # 2. Display cropped anatomical region
        if results.get("crop_image"):
            self._display_crop_image(results["crop_image"])

        # 3. Update predicted BCS score and certainty
        predicted_bcs = results["predicted_bcs"]
        certainty = results["confidence"]
        self.bcs_result_lbl.config(text=f"BCS: {predicted_bcs}")
        self.confidence_lbl.config(text=f"Prediction Certainty: {certainty}%")

        # 4. Enable actual ground truth inspection button
        self.btn_show_actual.config(state="normal")

        # 5. Format and display ordinal cumulative threshold probabilities
        thresholds = results.get("threshold_probabilities", {})
        lines = []
        for thresh_name, prob in thresholds.items():
            bar = "█" * int(prob // 10)
            lines.append(f"{thresh_name:<20}: {prob:5.1f}%  {bar}")
        self._set_prob_text("\n".join(lines))

    def _on_prediction_error(self, err_msg: str) -> None:
        """
        Handle inference failure, reset UI state, and present error dialog.

        Args:
            err_msg (str): Error message string to display.
        """
        self.btn_predict.config(state="normal", text="⚡ Run BCS Prediction")
        self.status_lbl.config(text="❌ Error", foreground="#f87171")
        messagebox.showerror("Error", f"An error occurred during inference:\n{err_msg}")


if __name__ == "__main__":
    app = OrdinalBCSApp()
    app.mainloop()
