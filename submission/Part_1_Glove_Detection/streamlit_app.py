import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import requests
from io import BytesIO
import time

# --- Configuration ---
st.set_page_config(
    page_title="Glove Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Dark Mode Styling ---
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Global Text Color */
    p, h1, h2, h3, h4, span, label {
        color: #e0e0e0 !important;
    }

    /* Metric Card Styling (Glassmorphism) */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #888 !important;
        font-size: 0.9rem !important;
        white-space: nowrap;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1c24;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0 0;
        padding: 8px 24px;
        color: #888;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 123, 255, 0.2);
        color: #007bff !important;
        border: 1px solid rgba(0, 123, 255, 0.3);
    }

    /* Tables & Dataframes */
    .stDataFrame, .stTable {
        background-color: #1a1c24 !important;
        color: #e0e0e0 !important;
    }

    /* Input Fields */
    .stTextInput input {
        background-color: #1a1c24 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    .title-box {
        text-align: center;
        padding: 40px 0;
        background: linear-gradient(90deg, #007bff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_image(image, model, conf, iou):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    start_time = time.time()
    results = model(img_bgr, conf=conf, iou=iou)[0]
    inference_time = (time.time() - start_time) * 1000
    
    annotated_img = img_array.copy()
    detections = []
    counts = {"glove_hand": 0, "bare_hand": 0}
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[cls_id]
        
        counts[label] = counts.get(label, 0) + 1
        
        detections.append({
            "Class": label.replace("_", " ").title(),
            "Confidence": f"{conf_score:.2%}",
            "Position": f"[{x1}, {y1}] -> [{x2}, {y2}]"
        })
        
        # Tech-focused box styling
        color = (46, 204, 113) if label == "glove_hand" else (231, 76, 60) # RGB (OpenCV uses BGR but we write to RGB array)
        # Flip color for OpenCV (BGR)
        color_bgr = (color[2], color[1], color[0])
        
        # Draw Box with glow
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 4)
        
        # Label with shadow
        l_text = f"{label.replace('_', ' ').upper()} {conf_score:.2f}"
        (w, h), _ = cv2.getTextSize(l_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
        cv2.rectangle(annotated_img, (x1, y1-h-15), (x1+w+10, y1), color, -1)
        cv2.putText(annotated_img, l_text, (x1+5, y1-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                    
    return annotated_img, detections, counts, inference_time

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2554/2554339.png", width=100)
    st.title("Glove Shield")
    st.markdown("---")
    
    st.header("⚙️ Model Controls")
    conf_val = st.slider("Confidence Level", 0.0, 1.0, 0.25, 0.05)
    iou_val = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    st.markdown("---")
    st.subheader("🛠️ Technical Stack")
    st.write("- **Engine:** YOLOv8n")
    st.write("- **Backend:** PyTorch/CPU")
    st.write("- **Dataset:** Hand-Glove v3")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.normpath(os.path.join(script_dir, "weights", "best.pt"))

# --- Main App ---
st.markdown("<div class='title-box'><h1>INDUSTRIAL GLOVE COMPLIANCE</h1></div>", unsafe_allow_html=True)

model = load_model(weights_path)

if model is None:
    st.error(f"❌ Critical Error: Weights not found at `{weights_path}`")
    st.stop()

# Layout
l_col, r_col = st.columns([1, 1], gap="large")

with l_col:
    st.subheader("📥 Input Interface")
    src_tabs = st.tabs(["📁 File Upload", "🔗 URL Link", "🧬 Dataset Samples"])
    
    img_final = None
    
    with src_tabs[0]:
        up = st.file_uploader("Drop protocol imagery here", type=["jpg", "jpeg", "png"])
        if up: img_final = Image.open(up).convert("RGB")
            
    with src_tabs[1]:
        u_input = st.text_input("Protocol Stream URL", placeholder="https://industrial.iot/camera1/frame.jpg")
        if u_input:
            try:
                res = requests.get(u_input, timeout=5)
                img_final = Image.open(BytesIO(res.content)).convert("RGB")
            except Exception as e: st.error(f"Network error: {e}")
                
    with src_tabs[2]:
        s_base = os.path.join(script_dir, "input_images")
        if os.path.exists(s_base):
            s_list = [f for f in os.listdir(s_base) if f.lower().endswith(".jpg")]
            sel = st.selectbox("Select factory frame", ["None"] + s_list)
            if sel != "None": img_final = Image.open(os.path.join(s_base, sel)).convert("RGB")
    
    if img_final:
        st.image(img_final, caption="Source Feed", use_container_width=True)
    else:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.2); padding: 60px; text-align: center; border-radius: 12px;'>
            <p style='color: #666; font-size: 1.2rem;'>Waiting for optical input...</p>
        </div>
        """, unsafe_allow_html=True)

with r_col:
    st.subheader("📡 AI Analytics")
    
    if img_final:
        with st.spinner("Processing safety telemetry..."):
            a_img, d_list, c_dict, i_ms = process_image(img_final, model, conf_val, iou_val)
        
        # Metrics
        met1, met2, met3 = st.columns(3)
        met1.metric("Latency", f"{i_ms:.1f}ms")
        met2.metric("Compliance", c_dict["glove_hand"])
        met3.metric("Alerts ⚠️", c_dict["bare_hand"], delta="REACHED" if c_dict["bare_hand"] > 0 else "NONE", delta_color="inverse")
        
        st.image(a_img, caption="AI Vision Protocol", use_container_width=True)
        
        # Download
        p_res = Image.fromarray(a_img)
        b = BytesIO()
        p_res.save(b, format="JPEG")
        st.download_button("💾 Extract Annotated Log", b.getvalue(), "safety_report.jpg", "image/jpeg", use_container_width=True)
    else:
        st.markdown("<div style='height: 400px; display: grid; place-items: center; color: #444;'>Analytics standby...</div>", unsafe_allow_html=True)

# --- Details ---
st.markdown("---")
st.subheader("📑 Protocol Documentation")

if img_final and d_list:
    det_cols = st.columns([3, 1])
    with det_cols[0]:
        st.dataframe(d_list, use_container_width=True)
    with det_cols[1]:
        st.markdown("""
        **Compliance Summary:**
        - Total Subjects: `{}`
        - Safe: `{}`
        - At Risk: `{}`
        """.format(len(d_list), c_dict["glove_hand"], c_dict["bare_hand"]))
elif img_final:
    st.info("No objects detected. Calibration may be required.")
else:
    st.write("Telemetry logs will be generated upon input.")
