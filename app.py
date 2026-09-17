import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Lab Portal", page_icon="🧪", layout="wide")
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
            # Using 'latin-1' encoding allows special lab characters (like µ or °) to pass through without crashing
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0) # reset file cursor
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"📊 Loaded '{uploaded_file.name}' successfully!")
        
        # 2. RUN THE MATH 
        # (Assuming your file has a column named 'Value' or 'Signal'. Change 'Value' to match your column headers)
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
            st.error("Error: Could not find a column named 'Value' in your uploaded file. Please make sure your column headers match.")
            st.write("Your column names are:", list(df.columns))
            
    except Exception as e:
        st.error(f"Could not parse file: {e}")
else:
    st.info("Please fill out metadata in the sidebar and upload a file to run calculations.")
