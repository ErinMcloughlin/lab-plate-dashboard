import streamlit as st
import os

st.set_page_config(page_title="Lab Portal", page_icon="🧪")
st.title("🧪 Lab Plate Upload Portal")

tech_name = st.text_input("Technologist Name")
plate_id = st.text_input("Plate / Batch ID")
uploaded_file = st.file_uploader("Upload Plate File", type=["csv", "xlsx", "txt"])

if st.button("Submit Plate"):
    if uploaded_file and tech_name and plate_id:
        os.makedirs("saved_plates", exist_ok=True)
        file_path = os.path.join("saved_plates", f"{plate_id}_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"🎉 Success! Plate {plate_id} has been securely saved.")
    else:
        st.warning("Please fill out all fields and upload a file first.")
