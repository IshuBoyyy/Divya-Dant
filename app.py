import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from PIL import Image

# 1. The Gaussian Math Formula
def perfect_curve(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

# --- UI SETUP ---
st.set_page_config(page_title="DivyaDant Diagnostics", page_icon="🦷", layout="centered")
st.title("🦷 DivyaDant Presentation Engine")
st.markdown("Upload a raw laser capture to generate the amplified optical profile.")

# --- APP LOGIC ---
uploaded_file = st.file_uploader("Select Capture", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Raw Sensor Capture", use_container_width=True)
    
    with st.spinner("Processing optical scattering & amplifying anatomical structures..."):
        img_array = np.array(image)
        
        # Auto-detect Laser Color
        if np.sum(img_array[:, :, 0]) > np.sum(img_array[:, :, 1]) * 1.5:
            channel, line_color, laser_name = img_array[:, :, 0], '#ef4444', 'Red Laser'
        else:
            channel, line_color, laser_name = img_array[:, :, 1], '#10b981', 'Green Laser'

        h, w = channel.shape

        # Extract Data
        target_y = np.unravel_index(np.argmax(channel, axis=None), channel.shape)[0]
        start_y, end_y = max(0, target_y - 15), min(h, target_y + 16)
        raw_intensities = np.mean(channel[start_y:end_y, :], axis=0)

        peak_idx = np.argmax(raw_intensities)
        x_data = np.arange(len(raw_intensities)) - peak_idx

        # --- THE EXAGGERATION ENGINE ---
        clean_real_data = savgol_filter(raw_intensities, window_length=51, polyorder=3)

        try:
            popt, _ = curve_fit(perfect_curve, x_data, clean_real_data, p0=[np.max(clean_real_data), 0, 50])
            base_curve = perfect_curve(x_data, *popt)
            width = int(abs(popt[2]) * 4) 
        except:
            base_curve = clean_real_data
            width = 0

        anatomical_bumps = clean_real_data - base_curve
        exaggerated_signal = base_curve + (anatomical_bumps * 3.0) 

        final_line = savgol_filter(exaggerated_signal, window_length=31, polyorder=3)
        final_line = np.clip(final_line, 0, 255)
        peak_intensity = np.max(final_line)

        # Auto-Diagnosis
        if width < 120:
            diagnosis, line_color = "POSITIVE: Severe Porosity (Signal Collapse)", '#ef4444'
        elif width > 220:
            diagnosis = "Healthy (Deep Translucency / Craze Lines)"
        else:
            diagnosis = "Healthy (Enamel Baseline)"

        # --- DRAW THE GRAPH ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x_data, final_line, color=line_color, linewidth=3)
        ax.set_xlim(-350, 350) 

        ax.set_title(f"DivyaDant Presentation Profile ({laser_name})", fontsize=16, color=line_color, pad=20)
        ax.set_xlabel("Distance from Impact Center (pixels)", fontsize=12)
        ax.set_ylabel("Laser Intensity (0-255)", fontsize=12)
        ax.set_ylim(0, 260)
        ax.grid(color='#334155', linestyle='--', linewidth=0.5)

        stats_text = (
            f"Calculated Base Width: ~{width} px\n"
            f"Peak Intensity: {peak_intensity:.1f} / 255\n"
            f"Diagnosis: {diagnosis}"
        )
        props = dict(boxstyle='round', facecolor='#1e293b', alpha=0.8, edgecolor=line_color)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, color='white', fontweight='bold')

        plt.tight_layout()
        
        # Display the graph in the app
        st.pyplot(fig)
        
        # Big, readable metrics for the phone screen
        st.success(f"**Final Diagnosis:** {diagnosis}")
        st.info(f"**Diagnostic Base Width:** {width} pixels")
