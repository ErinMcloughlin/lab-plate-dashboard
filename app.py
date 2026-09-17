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
            
    # COLUMN 3: Automation Deck (Linked to Venus Portal)
    with col3:
        st.subheader("🔬 Automation Deck")
        st.write("Direct integration matrix with the company's Hamilton Venus automation pipelines.")
        
        # This clean primary button handles everything cleanly by routing techs in a new window
        st.link_button(
            label="🌐 Open Venus Portal", 
            url="https://venus.harbinger-health.net/", 
            type="primary", 
            use_container_width=True
        )
        
        st.write("---")
        # Visual Helper Tag for the Technologists
        st.info("🔒 Secure Internal Subsystem. Click the action button above to sign in via the main network cluster.")


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


