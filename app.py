import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Lab Portal", page_icon="🧪", layout="wide")

# 1. INITIALIZE SESSION STATE ROUTING (Tracks which screen we are viewing)
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# ==========================================================
# SCREEN 1: THE WELCOME SCREEN / MAIN HUB (3 Columns)
# ==========================================================
if st.session_state.current_page == "home":
    st.title("🧪 Laboratory Command Center")
    st.write("Welcome to the Harbinger Health portal. Select a module below to begin your workflow.")
    st.write("---")
    
    # Create three equal-width columns side-by-side
    col1, col2, col3 = st.columns(3)
    
    # COLUMN 1: Plate Processing Tool
    with col1:
        st.subheader("📁 Plate Processing")
        st.write("Upload raw plate files and execute background blanking mathematics.")
        if st.button("🚀 Upload Plate", type="primary", use_container_width=True):
            st.session_state.current_page = "uploader"
            st.rerun()
            
    # COLUMN 2: Camera Inspection Tool
    with col2:
        st.subheader("📷 Visual Inspections")
        st.write("Trigger automated deck imagery, barcode scanning, or colony counts.")
        if st.button("🎥 Lab Cameras", type="primary", use_container_width=True):
            st.session_state.current_page = "cameras"
            st.rerun()
            
    # COLUMN 3: Placeholder for Future Tools
    with col3:
        st.subheader("🔬 Automation Deck")
        st.write("Future module space for direct liquid handler integrations and telemetry.")
        st.button("🔒 Locked Module", type="secondary", use_container_width=True, disabled=True)

# ==========================================================
# SCREEN 2: THE FILE UPLOAD & MATH SCREEN 
# ==========================================================
elif st.session_state.current_page == "uploader":
    # Global return button at the top
    if st.button("⬅️ Back to Main Hub"):
        st.session_state.current_page = "home"
        st.rerun()

    st.title("🧪 Lab Plate Upload & Math Analysis")
    
    # Sidebar inputs
    st.sidebar.header("Plate Metadata")
    tech_name = st.sidebar.text_input("Technologist Name")
    plate_id = st.sidebar.text_input("Plate / Batch ID")
    
    # User inputs a custom background blank value to subtract
    blank_value = st.sidebar.number_input("Negative Control / Blank Value to Subtract", value=0.0, step=0.1)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Plate File", type=["csv", "xlsx"])
    
    if uploaded_file and tech_name and plate_id:
        # 1. READ THE DATA WITH PANDAS (Smarter, robust encoding)
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0) # reset file cursor
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
            else:
                df = pd.read_excel(uploaded_file)
                
            st.success(f"📊 Loaded '{uploaded_file.name}' successfully!")
            
            # 2. RUN THE MATH 
            if 'Conc. [pg/µl]' in df.columns:
                # Simple math operation: Background subtraction
                df['Corrected_Value'] = df['Conc. [pg/µl]'] - blank_value
                
                # Descriptive Statistics
                mean_signal = df['Conc. [pg/µl]'].mean()
                std_signal = df['Conc. [pg/µl]'].std()
                max_signal = df['Conc. [pg/µl]'].max()
                
                # Display summary stats at the top of the dashboard
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Raw Signal", f"{mean_signal:.2f}")
                col2.metric("Standard Deviation", f"{std_signal:.2f}")
                col3.metric("Max Signal Observed", f"{max_signal:.2f}")
                
                # 3. DISPLAY THE PROCESSED TABLE TO THE TECH
                st.subheader("Processed Plate Data Table")
                st.dataframe(df)
                
                # 4. SAVE PROCESSED DATA TO DISK
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
    st.write("Streaming live imagery from internal automation deck network elements.")
    
    st.write("---")
    
    # Target URL of your lab camera's image/MJPEG video frame endpoint
    # Note: Most IP cameras serve a constant snapshot image or video feed at an endpoint 
    # like /snapshot.jpg, /image.jpg, /video, or /stream.mjpg. Update if yours requires a path!
    camera_url = "http://10.76.32.104"
    
    # 🎥 DISPLAY LIVE PREVIEW DIRECTLY ON SCREEN USING THE BACKEND RELAY
    try:
        # We tell the Python server to download the image box data directly over your building's LAN
        import urllib.request
        
        # Pull a clean snapshot directly from the camera
        with urllib.request.urlopen(camera_url, timeout=3) as response:
            image_bytes = response.read()
            
        # Draw the visual right onto your web link workspace page securely!
        st.image(image_bytes, caption="Live Automation Deck Capture (10.76.32.104)", use_container_width=True)
        
        # Provide an on-screen trigger button so technologists can refresh the stream snapshot manually
        if st.button("🔄 Refresh Live Feed View"):
            st.rerun()
            
    except Exception as e:
        st.error("❌ Unable to stream camera preview directly onto dashboard container.")
        st.warning("💡 To see this feed, ensure you are physically connected to the company Wi-Fi network and that the camera path is active.")
        st.write(f"Technical error context: {e}")

