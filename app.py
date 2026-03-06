import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- APP INTERFACE SETUP ---
st.set_page_config(page_title="ResoDent IRIS", page_icon="🦷", layout="centered")
st.title("🦷 ResoDent IRIS Mobile")
st.markdown("Upload a pitch-black laser capture to generate a smooth optical profile.")

# --- FILE UPLOADER (Taps into your phone's camera roll) ---
uploaded_file = st.file_uploader("Select Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read the image directly from the upload
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Capture", use_column_width=True)
    
    with st.spinner("Analyzing optical scattering..."):
        # Convert to numpy array (PIL opens as RGB)
        img_array = np.array(image)
        
        # 2. Isolate Green Channel (Index 1 in RGB)
        g_channel = img_array[:, :, 1]
        h, w = g_channel.shape
        
        # 3. Find Impact Center
        target_y = np.unravel_index(np.argmax(g_channel, axis=None), g_channel.shape)[0]
        
        # 4. Extract Horizontal Band
        start_y = max(0, target_y - 10)
        end_y = min(h, target_y + 11)
        band = g_channel[start_y:end_y, :]
        raw_intensities = np.mean(band, axis=0)
        
        # 5. Smooth the Graph
        window_size = 11
        smoothed_data = np.convolve(raw_intensities, np.ones(window_size)/window_size, mode='same')
        
        # 6. Calculate Metrics
        threshold = 25
        above_threshold = np.where(smoothed_data > threshold)[0]
        width = above_threshold[-1] - above_threshold[0] if len(above_threshold) > 0 else 0
        peak = np.max(smoothed_data)
        
        # --- GENERATE THE GRAPH ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 5))
        
        peak_idx = np.argmax(smoothed_data)
        x = np.arange(len(smoothed_data)) - peak_idx
        
        ax.plot(x, smoothed_data, color='#06b6d4', linewidth=3)
        ax.set_title("Optical Scattering Profile", color='#06b6d4')
        ax.set_xlabel("Distance (pixels)")
        ax.set_ylabel("Intensity (0-255)")
        ax.set_ylim(0, 260)
        ax.grid(color='#334155', linestyle='--', linewidth=0.5)
        
        # Add the data box
        stats = f"Base Width: ~{width} px\nPeak Intensity: {peak:.1f} / 255"
        props = dict(boxstyle='round', facecolor='#1e293b', alpha=0.8, edgecolor='#06b6d4')
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props, color='white')
        
        # Show the graph on the phone screen
        st.pyplot(fig)
        
        # Show big metrics for easy reading
        st.success(f"**Diagnostic Base Width:** {width} px")
