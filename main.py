import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/train/weights/best.pt"

model = YOLO(MODEL_PATH)

cap = None
webcam_running = False


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("Traffic Sign Detection System")
root.geometry("1100x800")
root.minsize(950, 700)
root.configure(bg="#f4f6f8")


# ============================================================
# COLORS
# ============================================================

BG = "#f4f6f8"
DARK = "#17202a"
BLUE = "#1976d2"
GREEN = "#2e7d32"
RED = "#c62828"
ORANGE = "#ef6c00"
WHITE = "#ffffff"
GRAY = "#607d8b"


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=DARK,
    height=130
)

header.pack(
    fill="x",
    padx=0,
    pady=0
)

title = tk.Label(
    header,
    text="TRAFFIC SIGN DETECTION",
    font=("Arial", 30, "bold"),
    fg=WHITE,
    bg=DARK
)

title.pack(pady=(20, 5))


subtitle = tk.Label(
    header,
    text="YOLO-Based Traffic Sign Recognition System",
    font=("Arial", 14),
    fg="#b0bec5",
    bg=DARK
)

subtitle.pack()


# ============================================================
# STATUS
# ============================================================

status_frame = tk.Frame(
    root,
    bg=BG
)

status_frame.pack(
    fill="x",
    pady=(15, 5)
)


status_label = tk.Label(
    status_frame,
    text="● READY",
    font=("Arial", 16, "bold"),
    fg=BLUE,
    bg=BG
)

status_label.pack()


# ============================================================
# IMAGE DISPLAY AREA
# ============================================================

image_frame = tk.Frame(
    root,
    bg=WHITE,
    highlightbackground="#cfd8dc",
    highlightthickness=2
)

image_frame.pack(
    padx=40,
    pady=15,
    fill="both",
    expand=True
)


image_label = tk.Label(
    image_frame,
    text="No image selected\n\nStart the webcam or upload an image",
    font=("Arial", 18),
    fg=GRAY,
    bg=WHITE
)

image_label.pack(
    fill="both",
    expand=True
)


# ============================================================
# DETECTION RESULT
# ============================================================

result_frame = tk.Frame(
    root,
    bg=BG
)

result_frame.pack(
    fill="x",
    pady=5
)


result_label = tk.Label(
    result_frame,
    text="Detection Result: None",
    font=("Arial", 18, "bold"),
    fg=DARK,
    bg=BG
)

result_label.pack()


# ============================================================
# SHOW IMAGE
# ============================================================

def show_image(frame):

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(frame_rgb)

    # Fit image inside display area
    image.thumbnail((850, 500))

    photo = ImageTk.PhotoImage(image)

    image_label.config(
        image=photo,
        text=""
    )

    image_label.image = photo


# ============================================================
# DETECT OBJECTS
# ============================================================

def process_frame(frame):

    results = model(
        frame,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()

    detections = []

    if len(results[0].boxes) > 0:

        boxes = results[0].boxes

        for i in range(len(boxes)):

            class_id = int(
                boxes.cls[i]
            )

            confidence = float(
                boxes.conf[i]
            )

            class_name = model.names[class_id]

            detections.append(
                (class_name, confidence)
            )

    return annotated_frame, detections


# ============================================================
# DISPLAY DETECTION
# ============================================================

def display_detection(detections):

    if not detections:

        result_label.config(
            text="Detection Result: No traffic sign detected",
            fg=ORANGE
        )

        status_label.config(
            text="● NO DETECTION",
            fg=ORANGE
        )

        return

    # Highest confidence detection
    best_detection = max(
        detections,
        key=lambda x: x[1]
    )

    class_name = best_detection[0]
    confidence = best_detection[1]

    result_label.config(
        text=f"Detected: {class_name}   |   Confidence: {confidence:.2%}",
        fg=GREEN
    )

    status_label.config(
        text="● DETECTION SUCCESSFUL",
        fg=GREEN
    )


# ============================================================
# UPLOAD IMAGE
# ============================================================

def upload_image():

    global webcam_running

    stop_webcam()

    file_path = filedialog.askopenfilename(
        title="Select Traffic Sign Image",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp"
            ),
            (
                "All Files",
                "*.*"
            )
        ]
    )

    if not file_path:
        return

    try:

        frame = cv2.imread(file_path)

        if frame is None:

            messagebox.showerror(
                "Error",
                "Unable to open the selected image."
            )

            return

        status_label.config(
            text="● PROCESSING IMAGE...",
            fg=BLUE
        )

        root.update_idletasks()

        annotated_frame, detections = process_frame(
            frame
        )

        show_image(
            annotated_frame
        )

        display_detection(
            detections
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            f"Image processing failed:\n\n{e}"
        )


# ============================================================
# WEBCAM LOOP
# ============================================================

def update_webcam():

    global cap
    global webcam_running

    if not webcam_running:
        return

    if cap is None:
        return

    ret, frame = cap.read()

    if not ret:

        status_label.config(
            text="● CAMERA ERROR",
            fg=RED
        )

        stop_webcam()

        return

    try:

        annotated_frame, detections = process_frame(
            frame
        )

        show_image(
            annotated_frame
        )

        display_detection(
            detections
        )

    except Exception as e:

        print(
            "Detection error:",
            e
        )

    if webcam_running:

        root.after(
            30,
            update_webcam
        )


# ============================================================
# START WEBCAM
# ============================================================

def start_webcam():

    global cap
    global webcam_running

    if webcam_running:
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        cap.release()
        cap = None

        messagebox.showerror(
            "Camera Error",
            "Could not open the webcam.\n\n"
            "Please check that your camera is available."
        )

        return

    webcam_running = True

    status_label.config(
        text="● WEBCAM RUNNING",
        fg=GREEN
    )

    result_label.config(
        text="Detection Result: Waiting for traffic sign...",
        fg=BLUE
    )

    update_webcam()


# ============================================================
# STOP WEBCAM
# ============================================================

def stop_webcam():

    global cap
    global webcam_running

    webcam_running = False

    if cap is not None:

        cap.release()
        cap = None

    status_label.config(
        text="● WEBCAM STOPPED",
        fg=BLUE
    )

    result_label.config(
        text="Detection Result: None",
        fg=DARK
    )


# ============================================================
# EXIT
# ============================================================

def exit_application():

    stop_webcam()

    root.destroy()


# ============================================================
# BUTTON AREA
# ============================================================

button_frame = tk.Frame(
    root,
    bg=BG
)

button_frame.pack(
    fill="x",
    padx=40,
    pady=15
)


# ============================================================
# BUTTON CREATOR
# ============================================================

def create_button(
    parent,
    text,
    command,
    bg_color
):

    return tk.Button(
        parent,
        text=text,
        command=command,
        font=("Arial", 13, "bold"),
        bg=bg_color,
        fg=WHITE,
        activebackground=bg_color,
        activeforeground=WHITE,
        relief="flat",
        bd=0,
        padx=15,
        pady=12,
        cursor="hand2"
    )


# ============================================================
# BUTTONS
# ============================================================

start_button = create_button(
    button_frame,
    "📷  Start Webcam",
    start_webcam,
    BLUE
)

start_button.pack(
    side="left",
    expand=True,
    fill="x",
    padx=5
)


stop_button = create_button(
    button_frame,
    "⛔  Stop Webcam",
    stop_webcam,
    ORANGE
)

stop_button.pack(
    side="left",
    expand=True,
    fill="x",
    padx=5
)


upload_button = create_button(
    button_frame,
    "🖼  Upload Image",
    upload_image,
    GREEN
)

upload_button.pack(
    side="left",
    expand=True,
    fill="x",
    padx=5
)


exit_button = create_button(
    button_frame,
    "❌  Exit",
    exit_application,
    RED
)

exit_button.pack(
    side="left",
    expand=True,
    fill="x",
    padx=5
)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Label(
    root,
    text="Powered by YOLO • Traffic Sign Recognition",
    font=("Arial", 10),
    fg=GRAY,
    bg=BG
)

footer.pack(
    pady=(0, 10)
)


# ============================================================
# WINDOW CLOSE HANDLER
# ============================================================

root.protocol(
    "WM_DELETE_WINDOW",
    exit_application
)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()