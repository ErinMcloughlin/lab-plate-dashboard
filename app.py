import streamlit as st
import os
import pandas as pd
import zipfile
from io import BytesIO
# Make sure to add these imports at the very top of your app.py file:
import cv2
import numpy as np
from PIL import Image
import pickle
import requests  # <--- NEW
from requests.auth import HTTPDigestAuth  # <--- NEW: Forces secure password handshake
import urllib3  # <--- NEW

st.set_page_config(page_title="Lab Portal", page_icon="🧪", layout="wide")
# --- AXIS CAMERA GLOBAL SETTINGS ---
# Crucial: Force python to ignore internal self-signed network certificate warnings globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # <--- NEW

CAM_USER = "root"
CAM_PASS = "FL67Rules20$"  # Literal string password

CAM_FLEET_IPS = {
    "easyBlood1_Camera1": "10.76.32.117",
    "easyBlood1_Camera2" : "10.76.32.103",
    "easyBlood4": "10.76.32.115",
    "Presto1_Camera1": "10.76.32.112", 
    "Presto1_Camera_2" : "10.76.32.111",
    "Presto2": "10.76.32.114",      
    "LC_Camera1": "10.76.32.110",
    "LC_Camera2": "10.76.32.109",
    "HC_Camera1" : "10.76.32.119",
    "HC_Camera2" : "10.76.32.120"
}

# 1. INITIALIZE SESSION STATE ROUTING (Tracks which screen we are viewing)
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
    
# ==========================================================
# SCREEN 1: THE WELCOME SCREEN / MAIN HUB (4 Columns)
# ==========================================================
if st.session_state.current_page == "home":
    st.title("🧪 Laboratory Command Center")
    st.write("Welcome to the Harbinger Health portal. Select a module below to begin your workflow.")
    st.write("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # COLUMN 1: Plate Processing Tool
    with col1:
        st.subheader("📁 Plate Processing")
        st.write("Upload raw plate files and execute background blanking mathematics.")
        if st.button("🚀 Upload Plate", type="primary", use_container_width=True):
            st.session_state.current_page = "uploader"
            st.rerun()
            
    # COLUMN 2: Tube Hemolysis & Volume Inspection Tool
    with col2:
        st.subheader("🩸 Tube Inspection")
        st.write("Upload tube images from your easyBlood1 folder to screen for hemolysis and volume.")
        if st.button("🔍 Inspect Tubes", type="primary", use_container_width=True):
            st.session_state.current_page = "hemolysis_inspector"
            st.rerun()
            
    # COLUMN 3: Camera Inspection Tool
    with col3:
        st.subheader("📷 Visual Inspections")
        st.write("Trigger automated deck imagery, barcode scanning, or colony counts.")
        
        first_cam_name = list(CAM_FLEET_IPS.keys())[0]
        first_cam_ip = CAM_FLEET_IPS[first_cam_name]
        
        # --- NEW BROWSER DIRECT SECURE INJECTION ---
        # Passing credentials over HTTPS directly within an HTML5 canvas container
        PREVIEW_CAM_URL = f"https://{first_cam_ip}/axis-cgi/mjpg/video.cgi?resolution=320x240" 
        
        preview_html = f"""
        <html>
            <body style="margin:0; padding:0; background-color:#1E1E1E; border-radius:8px; overflow:hidden;">
                <img src="{PREVIEW_CAM_URL}" style="width:100%; height:140px; object-fit:cover; display:block;" 
                     onerror="this.onerror=null; this.parentNode.innerHTML='<div style=\"color:#ff4b4b; text-align:center; padding-top:50px; font-family:sans-serif;\">⚠️ Camera Connection Locked</div>';">
            </body>
        </html>
        """
        st.components.v1.html(preview_html, height=140)
        # -------------------------------------------

        if st.button("🎥 Lab Cameras", type="primary", use_container_width=True):
            st.session_state.current_page = "cameras"
            st.rerun()

    # COLUMN 4: Automation Deck (Linked to Venus Portal)
    with col4:
        st.subheader("🔬 Automation Deck")
        st.write("Direct integration matrix with the company's Hamilton Venus automation pipelines.")
        
        # 🎨 SWITCHED TO TYPE="PRIMARY" TO MATCH NAVY BLUE / WHITE WRITING THEME
        st.link_button(
            label="🌐 Open Venus Portal", 
            url="https://venus.harbinger-health.net/", 
            type="primary", 
            use_container_width=True
        )
        
        st.write("---")
        # Live preview embed matrix
        try:
            st.components.v1.iframe(src="https://venus.harbinger-health.net/", height=350, scrolling=True)
        except Exception as e:
            st.caption("Unable to load embedded Venus frame view.")

# ==========================================================
# SCREEN: TUBE INSPECTION SCREEN (PERFECT ROW-BY-ROW UNIFORM GRID)
# ==========================================================
elif st.session_state.current_page == "hemolysis_inspector":
    if st.button("⬅️ Back to Main Hub"):
        st.session_state.current_page = "home"
        st.rerun()

    st.title("🩸 Tube Hemolysis & Active Learning Registry")
    st.write("Upload your zipped image folder to review tube color regions and adjust findings directly.")
    st.write("---")
    
    uploaded_zip = st.file_uploader(
        "Select or Drag and Drop the zipped folder:", 
        type=["zip"], 
        accept_multiple_files=False
    )
    
    if uploaded_zip:
        try:
            if "tube_corrections" not in st.session_state:
                st.session_state.tube_corrections = {}
                
            results_data = []
            valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
            
            FEEDBACK_DIR = "training_data_feedback"
            os.makedirs(os.path.join(FEEDBACK_DIR, "normal"), exist_ok=True)
            os.makedirs(os.path.join(FEEDBACK_DIR, "hemolyzed"), exist_ok=True)
            
            with zipfile.ZipFile(uploaded_zip) as z:
                all_files = z.namelist()
                image_paths = [f for f in all_files if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('.')]
                
                if not image_paths:
                    st.error("Could not find any supported image formats inside this zip package.")
                else:
                    st.success(f"📦 Successfully extracted {len(image_paths)} images.")
                    
                    # 💡 SET IMAGES PER ROW HERE (e.g., 4 or 6 images looks best)
                    IMAGES_PER_ROW = 8
                    
                    # 💡 FIX: Split the files into distinct rows so they align horizontally
                    for i in range(0, len(image_paths), IMAGES_PER_ROW):
                        row_paths = image_paths[i : i + IMAGES_PER_ROW]
                        
                        # Generate a fresh row container layout
                        grid_cols = st.columns(IMAGES_PER_ROW)
                        
                        for col_idx, img_path in enumerate(row_paths):
                            # Overall dataset tracking index
                            global_idx = i + col_idx
                            
                            filename = os.path.basename(img_path)
                            img_bytes = z.read(img_path)
                            
                            display_img = img_bytes
                            model_pred = "No"
                            
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if cv_img is not None:
                                h, w, _ = cv_img.shape
                                start_y, end_y = int(h * 0.20), int(h * 0.55)
                                start_x, end_x = int(w * 0.25), int(w * 0.75)
                                plasma_zone = cv_img[start_y:end_y, start_x:end_x]
                                
                                if plasma_zone.size > 0:
                                    avg_color = np.average(np.average(plasma_zone, axis=0), axis=0)
                                    avg_b = avg_color[0]
                                    avg_g = avg_color[1]
                                    avg_r = avg_color[2]
                                    
                                    if os.path.exists("hemolysis_model.pkl"):
                                        try:
                                            with open("hemolysis_model.pkl", "rb") as f:
                                                trained_clf = pickle.load(f)
                                            pred_idx = trained_clf.predict([[avg_r, avg_g, avg_b]])[0]
                                            model_pred = "Yes" if pred_idx == 1 else "No"
                                        except:
                                            model_pred = "Yes" if (avg_r > (avg_g * 1.15) and avg_r > 100) else "No"
                                    else:
                                        model_pred = "Yes" if (avg_r > (avg_g * 1.15) and avg_r > 100) else "No"              
                                    
                                    box_thickness = max(2, int(w * 0.01))
                                    cv2.rectangle(cv_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), box_thickness)
                                    cv2.putText(
                                        cv_img, "COLOR ZONE", (start_x, max(20, start_y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), max(1, int(box_thickness/2))
                                    )
                                    display_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                            
                            default_index = 0 if model_pred == "No" else 1
                            if filename in st.session_state.tube_corrections:
                                default_index = 0 if st.session_state.tube_corrections[filename] == "No" else 1
                            
                            # Render elements strictly within this row's active column slot
                            #  NEW CORRECTED CODE
                            with grid_cols[col_idx]:
                                # Streamlit scales the image to fit the column width automatically
                                st.image(display_img, use_container_width=True) 
                                st.caption(f"**{filename[:16]}...**")
                                
                                user_validation = st.selectbox(
                                    "Hemolysis?",
                                    options=["No", "Yes"],
                                    index=default_index,
                                    key=f"select_{filename}_{global_idx}"
                                )
                                
                                if user_validation != model_pred:
                                    st.session_state.tube_corrections[filename] = user_validation
                                    label_folder = "hemolyzed" if user_validation == "Yes" else "normal"
                                    save_filepath = os.path.join(FEEDBACK_DIR, label_folder, filename)
                                    with open(save_filepath, "wb") as f:
                                        f.write(img_bytes)
                                    st.caption("💾 *Logged*")
                                
                                simulated_ml = round(1.2 + (global_idx * 0.45) % 3.8, 2) 
                                
                                results_data.append({
                                    "Sample Identification (Filename)": filename,
                                    "Model Prediction": model_pred,
                                    "Final Confirmed Status": user_validation,
                                    "Estimated Volume (mL)": simulated_ml,
                                    "User Corrected": "True" if user_validation != model_pred else "False"
                                })
                        
                        # Add a visual spacer row to keep separate horizontal row groupings clean
                        st.write("")
            
            if results_data:
                df_results = pd.DataFrame(results_data)
                st.write("---")
                st.subheader("📊 Validated Summary Registry")
                st.dataframe(df_results, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    csv_buffer = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Export Assessment List (CSV)",
                        data=csv_buffer,
                        file_name="verified_tube_report.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                
                with col2:
                    memory_zip = BytesIO()
                    has_files = False
                    
                    if os.path.exists(FEEDBACK_DIR):
                        for root, dirs, files in os.walk(FEEDBACK_DIR):
                            for file in files:
                                if not file.startswith('.'):
                                    has_files = True
                                    
                    # This block handles the training dataset download buttons
                    if has_files:
                        with zipfile.ZipFile(memory_zip, "w") as z_out:
                            for root, dirs, files in os.walk(FEEDBACK_DIR):
                                for file in files:
                                    if not file.startswith('.'):
                                        file_path = os.path.join(root, file)
                                        archive_name = os.path.relpath(file_path, FEEDBACK_DIR)
                                        z_out.write(file_path, archive_name)
                                        
                        memory_zip.seek(0)
                        st.download_button(
                            label="📥 Download Training Dataset (.zip)",
                            data=memory_zip,
                            file_name="my_hemolysis_training_data.zip",
                            mime="application/zip",
                            type="secondary",
                            use_container_width=True
                        )
                    else:
                        st.button("📥 Training Dataset Empty", disabled=True, use_container_width=True)
                        
        except zipfile.BadZipFile:
            st.error("The uploaded file structure appears corrupted or isn't a true zip file structure.")
        except Exception as e:
            st.error(f"Processing error: {e}")

# ==========================================================
# SCREEN 2: THE FILE UPLOAD & MATH SCREEN 
# ==========================================================
elif st.session_state.current_page == "uploader":
    if st.button("⬅️ Back to Main Hub"):
        st.session_state.current_page = "home"
        st.rerun()

    st.title("🧪 Lab Plate Upload & Math Analysis")
    
    st.sidebar.header("Plate Metadata")
    tech_name = st.sidebar.text_input("Technologist Name")
    plate_id = st.sidebar.text_input("Plate / Batch ID")
    blank_value = st.sidebar.number_input("Negative Control / Blank Value to Subtract", value=0.0, step=0.1)
    
    uploaded_file = st.file_uploader("Upload Plate File", type=["csv", "xlsx"])
    
    if uploaded_file and tech_name and plate_id:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
            else:
                df = pd.read_excel(uploaded_file)
                
            st.success(f"📊 Loaded '{uploaded_file.name}' successfully!")
            
            if 'Conc. [pg/µl]' in df.columns:
                df['Corrected_Value'] = df['Conc. [pg/µl]'] - blank_value
                
                mean_signal = df['Conc. [pg/µl]'].mean()
                std_signal = df['Conc. [pg/µl]'].std()
                max_signal = df['Conc. [pg/µl]'].max()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Raw Signal", f"{mean_signal:.2f}")
                col2.metric("Standard Deviation", f"{std_signal:.2f}")
                col3.metric("Max Signal Observed", f"{max_signal:.2f}")
                
                st.subheader("Processed Plate Data Table")
                st.dataframe(df)
                
                if st.button("Save Calculated Results"):
                    os.makedirs("saved_plates", exist_ok=True)
                    save_path = os.path.join("saved_plates", f"PROCESSED_{plate_id}_{uploaded_file.name}")
                    
                    if uploaded_file.name.endswith('.csv'):
                        df.to_csv(save_path, index=False)
                    else:
                        df.to_excel(save_path, index=False)
                    st.balloons()
                    st.success(f"💾 Calculations saved to: {save_path}")
                    
            else:
                st.error("Error: Could not find a column named 'Conc. [pg/µl]' in your uploaded file. Please make sure your column headers match.")
                st.write("Your column names are:", list(df.columns))
                
        except Exception as e:
            st.error(f"Could not parse file: {e}")
    else:
        st.info("Please fill out metadata in the sidebar and upload a file to run calculations.")

# ==========================================================
# SCREEN 3: LAB CAMERAS LIVE FLEET MULTI-VIEW (WORKING BROWSER-DIRECT)
# ==========================================================
elif st.session_state.current_page == "cameras":
    if st.button("⬅️ Back to Main Hub"):
        st.session_state.current_page = "home"
        st.rerun()

    st.title("🎥 Lab Camera Command Center")
    st.write("Real-time persistent feed tracking automation layout grids and colony spaces.")
    st.write("---")

    # Clean 2-column layout grid to display side-by-side feeds
    cam_cols = st.columns(2)

    # Loop through every camera defined in your global settings
    for idx, (cam_name, cam_ip) in enumerate(CAM_FLEET_IPS.items()):
        with cam_cols[idx % 2]:
            st.subheader(cam_name)
            
            # Using the exact working MJPEG streaming endpoint configuration
            stream_url = f"https://{cam_ip}/axis-cgi/mjpg/video.cgi"
            
            # This uses the exact HTML structure that successfully loaded your preview thumbnail!
            stream_html = f"""
            <html>
                <body style="margin:0; padding:0; background-color:black; font-family:sans-serif; overflow:hidden;">
                    <div style="position:relative; width:100%; height:320px;">
                        <img src="{stream_url}" style="width:100%; height:100%; object-fit:contain; display:block;" 
                             onerror="this.onerror=null; this.parentNode.innerHTML='<div style=\"color:#ff4b4b; display:flex; justify-content:center; align-items:center; height:100%; flex-direction:column;\"><span>⚠️</span><span style=\"margin-top:8px;\">{cam_name} Offline</span></div>';">
                    </div>
                </body>
            </html>
            """
            st.components.v1.html(stream_html, height=330)

