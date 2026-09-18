import streamlit as st
import os
import pandas as pd
import zipfile
from io import BytesIO
# Make sure to add these imports at the very top of your app.py file:
import cv2
import numpy as np
from PIL import Image

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
            
    # COLUMN 2: Tube Hemolysis Inspection Tool
    with col2:
        st.subheader("🩸 Tube Inspection")
        st.write("Upload tube images from your easyBlood1 folder to screen for hemolysis.")
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
            url="https://venus.harbinger-health.net/", 
            type="primary", 
            use_container_width=True
        )
        
        st.write("---")
        try:
            st.components.v1.iframe(src="https://harbinger-health.net", height=350, scrolling=True)
        except Exception as e:
            st.caption("Unable to load embedded Venus frame view.")


# ==========================================================
# SCREEN: TUBE INSPECTION SCREEN (ACTIVE LEARNING FEEDBACK)
# ==========================================================
elif st.session_state.current_page == "hemolysis_inspector":
    if st.button("⬅️ Back to Main Hub"):
        st.session_state.current_page = "home"
        st.rerun()

    st.title("🩸 Tube Hemolysis & Active Learning Registry")
    st.write("Upload your zipped image folder. Correct any mistakes below to train the model to be smarter.")
    
    st.info(
        "💡 **Active Learning Active:** Adjusting the dropdown menus below automatically "
        "logs corrections to your local training dataset."
    )
    st.write("---")
    
    uploaded_zip = st.file_uploader(
        "Select or Drag and Drop the zipped folder:", 
        type=["zip"], 
        accept_multiple_files=False
    )
    
    if uploaded_zip:
        try:
            # Initialize a session state dictionary to hold user corrections if it doesn't exist
            if "tube_corrections" not in st.session_state:
                st.session_state.tube_corrections = {}
                
            results_data = []
            valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
            
            # Directory where corrected data will be stored for retraining
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
                    
                    grid_cols = st.columns(6) # Slightly wider columns to fit the selectbox comfortably
                    
                    for idx, img_path in enumerate(image_paths):
                        filename = os.path.basename(img_path)
                        img_bytes = z.read(img_path)
                    # --- 1. MODEL PREDICTION (Active Learning Engine) ---
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if cv_img is not None:
                        h, w, _ = cv_img.shape
                        start_y, end_y = int(h * 0.20), int(h * 0.55)
                        start_x, end_x = int(w * 0.25), int(w * 0.75)
                        plasma_zone = cv_img[start_y:end_y, start_x:end_x]
                        avg_color = np.average(np.average(plasma_zone, axis=0), axis=0)
                        avg_b, avg_g, avg_r = avg_color[0], avg_color[1], avg_color[2]
                        
                        # Check if a personalized trained model file exists in your repository
                        if os.path.exists("hemolysis_model.pkl"):
                            try:
                                with open("hemolysis_model.pkl", "rb") as f:
                                    trained_clf = pickle.load(f)
                                # Use machine learning prediction (0 = No, 1 = Yes)
                                pred_idx = trained_clf.predict([[avg_r, avg_g, avg_b]])[0]
                                model_pred = "Yes" if pred_idx == 1 else "No"
                            except:
                                # Emergency fallback if model file fails to read
                                model_pred = "Yes" if (avg_r > (avg_g * 1.15) and avg_r > 100) else "No"
                        else:
                            # Standard fallback rule until you run train_model.py the first time
                            model_pred = "Yes" if (avg_r > (avg_g * 1.15) and avg_r > 100) else "No"              
                        # --- 2. INTERACTIVE USER FEEDBACK TRACKING ---
                        # Use the previously saved correction if the user already changed it during this session
                        default_index = 0 if model_pred == "No" else 1
                        if filename in st.session_state.tube_corrections:
                            default_index = 0 if st.session_state.tube_corrections[filename] == "No" else 1
                        
                        with grid_cols[idx % 6]:
                            st.image(display_img, use_container_width=True)
                            st.caption(f"**{filename[:12]}...**")
                            
                            # Interactive Dropdown directly below the image preview card
                            user_validation = st.selectbox(
                                "Hemolysis?",
                                options=["No", "Yes"],
                                index=default_index,
                                key=f"select_{filename}_{idx}"
                            )
                            
                            # If user changes the value away from the original prediction, log it
                            if user_validation != model_pred:
                                st.session_state.tube_corrections[filename] = user_validation
                                
                                # Write the corrected image file out to the feedback directory
                                label_folder = "hemolyzed" if user_validation == "Yes" else "normal"
                                save_filepath = os.path.join(FEEDBACK_DIR, label_folder, filename)
                                with open(save_filepath, "wb") as f:
                                    f.write(img_bytes)
                                    
                                st.caption("💾 *Logged to Training Data*")
                            
                            # Final evaluation value to save in the export table
                            final_decision = user_validation
                            
                            results_data.append({
                                "Sample Identification (Filename)": filename,
                                "Model Prediction": model_pred,
                                "Final Confirmed Status": final_decision,
                                "User Corrected": "True" if user_validation != model_pred else "False"
                            })
            
            if results_data:
                df_results = pd.DataFrame(results_data)
                st.write("---")
                st.subheader("📊 Validated Summary Registry")
                st.dataframe(df_results, use_container_width=True)
                
                # Check metrics of corrections made
                corrections_count = df_results[df_results["User Corrected"] == "True"].shape[0]
                if corrections_count > 0:
                    st.toast(f"Logged {corrections_count} sample corrections to disk during this session!", icon="💾")
                
                csv_buffer = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Assessment List (CSV)",
                    data=csv_buffer,
                    file_name="verified_tube_report.csv",
                    mime="text/csv",
                    type="primary"
                )
                # Place this right below your CSV st.download_button block inside app.py:

                st.write("---")
                st.subheader("⚙️ Active Learning Admin Panel")
                st.write("Download your collected training image data package to retrain your model on your computer.")
                
                # Create an in-memory zip file of the feedback folders
                memory_zip = BytesIO()
                has_files = False
                
                with zipfile.ZipFile(memory_zip, "w") as z_out:
                    FEEDBACK_DIR = "training_data_feedback"
                    if os.path.exists(FEEDBACK_DIR):
                        for root, dirs, files in os.walk(FEEDBACK_DIR):
                            for file in files:
                                if not file.startswith('.'):
                                    file_path = os.path.join(root, file)
                                    # Maintain the subfolder structure (normal/ vs hemolyzed/) inside the zip package
                                    archive_name = os.path.relpath(file_path, FEEDBACK_DIR)
                                    z_out.write(file_path, archive_name)
                                    has_files = True
                
                if has_files:
                    memory_zip.seek(0)
                    st.download_button(
                        label="📥 Download Training Dataset (.zip)",
                        data=memory_zip,
                        file_name="my_hemolysis_training_data.zip",
                        mime="application/zip",
                        type="secondary"
                    )
                else:
                    st.info("ℹ️ No human corrections have been logged yet. Changes will appear here once you correct a sample!")

        except zipfile.BadZipFile:
            st.error("The uploaded file structure appears corrupted or isn't a true zip file structure.")
        except Exception as e:
            st.error(f"Processing error: {e}")
    else:
        st.warning("Please upload the `easyBlood1 Images.zip` archive file to execute analytical mapping.")

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
