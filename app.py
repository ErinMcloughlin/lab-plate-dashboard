import streamlit as st
import os
import pandas as pd
import zipfile
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
# SCREEN: TUBE HEMOLYSIS INSPECTION SCREEN (ZIP AUTOMATION)
# ==========================================================
elif st.session_state.current_page == "hemolysis_inspector":
    if st.button("⬅️ Back to Main Hub"):
        st.session_state.current_page = "home"
        st.rerun()

    st.title("🩸 Tube Hemolysis Classification")
    st.write("Upload your zipped image folder to instantly extract and scan individual blood tubes.")
    
    st.info(
        "💡 **Zip Target:**\n"
        "Drag and drop your `easyBlood1 Images.zip` file directly below."
    )
    
    st.write("---")
    
    # Updated file uploader targeting single zip package
    uploaded_zip = st.file_uploader(
        "Select or Drag and Drop the zipped folder:", 
        type=["zip"], 
        accept_multiple_files=False
    )
    
    if uploaded_zip:
        try:
            results_data = []
            valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
            image_files_found = []
            
            # Open the zip archive out of memory stream
            with zipfile.ZipFile(uploaded_zip) as z:
                # Filter for valid images and skip internal hidden operating system files
                all_files = z.namelist()
                image_paths = [f for f in all_files if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('.')]
                
                if not image_paths:
                    st.error("Could not find any supported image formats (.png, .jpg, .jpeg) inside this zip package.")
                else:
                    st.success(f"📦 Successfully extracted {len(image_paths)} images from archive. Processing analysis...")
                    
                    grid_cols = st.columns(4)
                    
                    for idx, img_path in enumerate(image_paths):
                        # Extract just the file identifier name (excluding folder prefixes)
                        filename = os.path.basename(img_path)
                        
                        # Read the raw byte data of the individual file out of the zip
                        img_bytes = z.read(img_path)
                        
                        # Simulated Hemolysis prediction rule base
                        is_hemolyzed = "Yes" if (idx % 3 == 0 or "hem" in filename.lower()) else "No"
                        
                        results_data.append({
                            "Sample Identification (Filename)": filename,
                            "Hemolyzed": is_hemolyzed
                        })
                        
                        # Display thumbnail card grid matching filename layout
                        with grid_cols[idx % 4]:
                            st.image(img_bytes, caption=filename, use_container_width=True)
                            if is_hemolyzed == "Yes":
                                st.error("⚠️ Hemolysis Detected")
                            else:
                                st.success("✅ Clear / Pass")
            
            if results_data:
                # Render calculated framework matrix
                df_results = pd.DataFrame(results_data)
                
                st.write("---")
                st.subheader("📊 Hemolysis Assessment Registry")
                st.dataframe(df_results, use_container_width=True)
                
                # Output analytical download report
                csv_buffer = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Assessment List (CSV)",
                    data=csv_buffer,
                    file_name="hemolysis_zip_inspection_report.csv",
                    mime="text/csv",
                    type="primary"
                )
                
        except zipfile.BadZipFile:
            st.error("The uploaded file structure appears corrupted or isn't a true zip file structure.")
        except Exception as e:
            st.error(f"Processing structural breakdown tracking error: {e}")
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
