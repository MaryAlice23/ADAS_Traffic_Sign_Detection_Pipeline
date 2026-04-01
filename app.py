import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from ultralytics import YOLO

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")   # ✅ No auth, auto-downloads once

yolo_model = load_yolo()


def detect_with_yolo(image):
    results = yolo_model(image)

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return None, None

        # Get best box (highest confidence)
        best_box = max(boxes, key=lambda b: float(b.conf))

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        cropped = image[y1:y2, x1:x2]

        # Safety check
        if cropped.size == 0:
            return None, None

        cropped = cv2.resize(cropped, (64, 64))

        return cropped, (x1, y1, x2-x1, y2-y1)

    return None, None
# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="ADAS Traffic Sign Detection", layout="wide")

st.title("🚗 ADAS Traffic Sign Detection & Classification")
st.write("Step 1: Detect sign → Step 2: Classify into 43 categories")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_my_model():
    model = load_model("resnet_model.h5", compile=False)  # 🔁 Use YOUR best CNN model
    return model

model = load_my_model()

# -------------------------------
# CLASS LABELS
# -------------------------------
# Load mapping (same as training)
sign_names = {
0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
9:'No passing', 10:'No passing veh > 3.5 tons', 11:'Right-of-way at intersection',
12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on right',
25:'Road work', 26:'Traffic signals', 27:'Pedestrians',
28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow',
31:'Wild animals crossing', 32:'End speed + passing limits',
33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only',
36:'Go straight or right', 37:'Go straight or left',
38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}
class_indices = {
'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6,
'15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12,
'20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18,
'26': 19, '27': 20, '28': 21, '29': 22, '3': 23, '30': 24,
'31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30,
'37': 31, '38': 32, '39': 33, '4': 34, '40': 35, '41': 36,
'42': 37, '5': 38, '6': 39, '7': 40, '8': 41, '9': 42
}

# Reverse mapping
index_to_class = {v: int(k) for k, v in class_indices.items()}

# -------------------------------
# DETECTION FUNCTION
# -------------------------------
def detect_and_crop_sign(image):
    import cv2
    import numpy as np

    img = image.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # 🔴 RED MASK
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

    # -------------------------
    # 🔵 BLUE MASK
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # -------------------------
# 🟡 YELLOW MASK (NEW 🔥)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # -------------------------
    # ⚪ WHITE / GRAY MASK (NEW FIX)
    mask_white = cv2.inRange(gray, 170, 255)
    edges = cv2.Canny(gray, 50, 150)

    # -------------------------
    # 🔥 COMBINE ALL
    mask = mask_red + mask_blue + mask_yellow + mask_white + edges

    # Clean noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    best = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if 0.4 < aspect_ratio < 1.6:
            if area > max_area:
                best = (x, y, w, h)
                max_area = area

    if best is None:
        return None, None

    x, y, w, h = best
    cropped = img[y:y+h, x:x+w]
    cropped = cv2.resize(cropped, (64, 64))

    return cropped, best
# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------
def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload road scene image", type=["jpg","png","jpeg"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    original = img.copy()

    crop, bbox = detect_and_crop_sign(img)
    if crop is None:
        crop, bbox = detect_with_yolo(img)
        
     

# 🔁 Fallback 2: CENTER CROP (LAST RESORT)
    if crop is None:
        h, w, _ = img.shape
        crop = img[h//4:3*h//4, w//4:3*w//4]
        crop = cv2.resize(crop, (64,64))
        bbox = (w//4, h//4, w//2, h//2)
        # tighter crop (reduce background noise)
        h, w, _ = crop.shape
        crop = crop[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)]
        crop = cv2.resize(crop, (64,64))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original Image")
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))

    if crop is not None:

        x, y, w, h = bbox
        boxed = original.copy()
        cv2.rectangle(boxed, (x,y), (x+w, y+h), (0,255,0), 2)

        with col2:
            st.subheader("🎯 Detected Sign")
            st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB))

        # Convert to RGB BEFORE prediction
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        with col3:
            st.subheader("✂️ Cropped Sign")
            st.image(crop_rgb)
            
            
            

        # Prediction
        # -------------------------------
# PREDICTION (FIXED)
# -------------------------------
        # sharpen image (VERY IMPORTANT for symbols)
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

        crop = cv2.filter2D(crop, -1, kernel)
# Convert BGR → RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

# Resize (safety)
        crop_resized = cv2.resize(crop_rgb, (64, 64))

# Normalize
        crop_norm = crop_resized.astype("float32") / 255.0

# Add batch dimension
        crop_input = np.expand_dims(crop_norm, axis=0)

# Predict
        pred = model.predict(crop_input)

        class_id = np.argmax(pred)
        confidence = np.max(pred)

# Mapping fix
        true_class_id = index_to_class[class_id]
        final_label = sign_names[true_class_id]

        st.success(f"Prediction: {final_label}")
        st.info(f"Confidence: {confidence:.2f}")
        

        # Top 3 predictions
        st.subheader("🔍 Top 3 Predictions")

        top3 = pred[0].argsort()[-3:][::-1]

        for i in top3:
            true_class_id = index_to_class[i]   # FIX mapping
            label = sign_names[true_class_id]
            st.write(f"{label} : {pred[0][i]:.2f}")

    else:
        st.error("❌ No traffic sign detected. Try another image.")