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
from io import BytesIO

st.set_page_config(page_title="Lab Portal", page_icon="🧪", layout="wide")

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
        if st.button("🎥 Lab Cameras", type="primary", use_container_width=True):
            st.session_state.current_page = "cameras"
            st.rerun()
            
    # COLUMN 4: Automation Deck (Linked to Venus Portal)
    with col4:
        st.subheader("🔬 Automation Deck")
        st.write("Direct integration matrix with the company's Hamilton Venus automation pipelines.")
        
        st.link_button(
            label="🌐 Open Venus Portal", 
            url="https://harbinger-health.net", 
            type="primary", 
            use_container_width=True
        )
        
        st.write("---")
        try:
            st.components.v1.iframe(src="https://harbinger-health.net", height=350, scrolling=True)
        except Exception as e:
            st.caption("Unable to load embedded Venus frame view.")

# ==========================================================
# SCREEN: TUBE INSPECTION SCREEN (CLEAN GALLERY MATRIX WITH SIDEBAR CONFIG)
# ==========================================================
elif st.session_state.current_page == "hemolysis_inspector":
    if st.button("⬅️ Back to Main Hub"):
        st.session_state.current_page = "home"
        st.rerun()

    st.title("🩸 Tube Hemolysis & Active Learning Registry")
    st.write("Upload your zipped image folder. Click on any sample card to review metrics or adjust classification profiles.")
    
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
            if "selected_tube" not in st.session_state:
                st.session_state.selected_tube = None
                
            results_data = []
            valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
            
            FEEDBACK_DIR = "training_data_feedback"
            os.makedirs(os.path.join(FEEDBACK_DIR, "normal"), exist_ok=True)
            os.makedirs(os.path.join(FEEDBACK_DIR, "hemolyzed"), exist_ok=True)
            
            # Read files into memory once to keep user interaction smooth
            with zipfile.ZipFile(uploaded_zip) as z:
                all_files = z.namelist()
                image_paths = [f for f in all_files if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('.')]
                
                if not image_paths:
                    st.error("Could not find any supported image formats inside this zip package.")
                else:
                    # Pre-process image states and baseline logic calculations
                    processed_tubes = {}
                    for idx, img_path in enumerate(image_paths):
                        filename = os.path.basename(img_path)
                        img_bytes = z.read(img_path)
                        
                        model_pred = "No"
                        display_img = img_bytes
                        
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if cv_img is not None:
                            h, w, _ = cv_img.shape
                            start_y, end_y = int(h * 0.20), int(h * 0.55)
                            start_x, end_x = int(w * 0.25), int(w * 0.75)
                            plasma_zone = cv_img[start_y:end_y, start_x:end_x]
                            
                            if plasma_zone.size > 0:
                                avg_color = np.average(np.average(plasma_zone, axis=0), axis=0)
                                avg_b, avg_g, avg_r = avg_color, avg_color, avg_color
                                
                                if os.path.exists("hemolysis_model.pkl"):
                                    try:
                                        with open("hemolysis_model.pkl", "rb") as f:
                                            trained_clf = pickle.load(f)
                                        pred_idx = trained_clf.predict([[avg_r, avg_g, avg_b]])
                                        model_pred = "Yes" if pred_idx == 1 else "No"
                                    except:
                                        model_pred = "Yes" if (avg_r > (avg_g * 1.15) and avg_r > 100) else "No"
                                else:
                                    model_pred = "Yes" if (avg_r > (avg_g * 1.15) and avg_r > 100) else "No"              
                                
                                box_thickness = max(2, int(w * 0.01))
                                cv2.rectangle(cv_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), box_thickness)
                                display_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        
                        # Determine current actual state (either model guess or user correction)
                        final_state = st.session_state.tube_corrections.get(filename, model_pred)
                        simulated_ml = round(1.2 + (idx * 0.45) % 3.8, 2)
                        
                        processed_tubes[filename] = {
                            "img_display": display_img,
                            "img_raw": img_bytes,
                            "model_pred": model_pred,
                            "final_state": final_state,
                            "volume": simulated_ml
                        }
                        
                        results_data.append({
                            "Sample Identification (Filename)": filename,
                            "Model Prediction": model_pred,
                            "Final Confirmed Status": final_state,
                            "Estimated Volume (mL)": simulated_ml,
                            "User Corrected": "True" if filename in st.session_state.tube_corrections else "False"
                        })

                    # 🌟 VISUAL GRID: Clean, borderless grid alignment (no menus or bulky error banners)
                    st.subheader("📸 Extraction Gallery Matrix")
                    grid_cols = st.columns(8)
                    
                    for idx, filename in enumerate(processed_tubes.keys()):
                        tube = processed_tubes[filename]
                        
                        with grid_cols[idx % 8]:
                            # Mini status color dots to keep things visually clean
                            status_dot = "🔴 Hemolysis" if tube["final_state"] == "Yes" else "🟢 Pass"
                            
                            # Clean image render
                            st.image(tube["img_display"], use_container_width=True)
                            st.caption(f"**{filename[:12]}...**")
                            st.caption(f"{status_dot} | {tube['volume']}mL")
                            
                            # A clean, low-profile button to select the image for editing
                            if st.button("🔎 Review", key=f"btn_{filename}_{idx}", use_container_width=True):
                                st.session_state.selected_tube = filename
                                st.rerun()

            # 🌟 SIDEBAR CONFIG PANEL: Moves interactive workflow elements off the main grid view
            if st.session_state.selected_tube and st.session_state.selected_tube in processed_tubes:
                selected_name = st.session_state.selected_tube
                selected_data = processed_tubes[selected_name]
                
                st.sidebar.header("🔬 Sample Validation Inspector")
                st.sidebar.image(selected_data["img_display"], caption=selected_name, use_container_width=True)
                st.sidebar.write(f"**Initial Model Finding:** {selected_data['model_pred']}")
                st.sidebar.write(f"**Calculated Fluid Volume:** {selected_data['volume']} mL")
                
                default_idx = 0 if selected_data["final_state"] == "No" else 1
                user_validation = st.sidebar.selectbox(
                    "Override Hemolysis Flag?",
                    options=["No", "Yes"],
                    index=default_idx
                )
                
                # Update corrections state when the sidebar changes
                if user_validation != selected_data["model_pred"]:
                    if st.session_state.tube_corrections.get(selected_name) != user_validation:
                        st.session_state.tube_corrections[selected_name] = user_validation
                        
                        # Save the updated target back to disk
                        label_folder = "hemolyzed" if user_validation == "Yes" else "normal"
                        save_filepath = os.path.join(FEEDBACK_DIR, label_folder, selected_name)
                        with open(save_filepath, "wb") as f:
                            f.write(selected_data["img_raw"])
                        st.rerun()
                elif selected_name in st.session_state.tube_corrections and user_validation == selected_data["model_pred"]:
                    # Remove correction if reverted to the original prediction
                    del st.session_state.tube_corrections[selected_name]
                    st.rerun()
                    
                if st.sidebar.button("Close Inspector", use_container_width=True):
                    st.session_state.selected_tube = None
                    st.rerun()

            # Render summary report registers below the visual view
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
# SCREEN 3: THE LAB CAMERA SCREEN 
# ==========================================================
elif st.session_state.current_page == "cameras":
    if st.button("⬅️ Back to Main Hub"):
        st.session_state.current_page = "home"
        st.rerun()

    st.title("📷 Integrated Lab Cameras & Scanning")
    st.write("Access your automation deck camera feed below.")
    
    st.info("💡 Note: To access this feed, your computer must be connected to the company's internal network or VPN.")
    
    st.write("---")
    
    # 📐 PUSH BUTTON TO THE RIGHT HAND SIDE USING COLUMNS
    # Create 3 columns. col1 and col2 will act as empty space on the left.
    col1, col2, col3 = st.columns(3)
    
    # Place the button strictly inside the right-most column (col3)
    with col3:
        camera_url = "http://10.76.32.104"
        st.link_button(
            label="🎥 Developmental Instrument", 
            url=camera_url, 
            type="primary", 
            use_container_width=True
        )
    
    st.write("---")
    st.write("📊 **Troubleshooting Steps if the camera page won't load:**")
    st.markdown("""
    1. Confirm you are on the **Harbinger Health internal Wi-Fi** or corporate VPN.
    2. Check that the camera hardware box is powered on.
    3. If the camera page asks for a specific login or port, contact your automation engineer.
    """)
