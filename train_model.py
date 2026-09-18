import os
import cv2
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

def train_new_model():
    FEEDBACK_DIR = "training_data_feedback"
    categories = ["normal", "hemolyzed"]
    
    X = [] # Holds the color features
    y = [] # Holds the matching labels (0 for normal, 1 for hemolyzed)
    
    print("🔄 Scanning feedback folders for human-corrected data...")
    
    for label_idx, category in enumerate(categories):
        folder_path = os.path.join(FEEDBACK_DIR, category)
        if not os.path.exists(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            # Skip hidden system files
            if filename.startswith('.'):
                continue
                
            # Read image
            cv_img = cv2.imread(img_path)
            if cv_img is None:
                continue
                
            # Focus on the exact same plasma zone we use in the dashboard
            h, w, _ = cv_img.shape
            start_y, end_y = int(h * 0.20), int(h * 0.55)
            start_x, end_x = int(w * 0.25), int(w * 0.75)
            plasma_zone = cv_img[start_y:end_y, start_x:end_x]
            
            # Extract average RGB intensities
            avg_color = np.average(np.average(plasma_zone, axis=0), axis=0)
            avg_b, avg_g, avg_r = avg_color[0], avg_color[1], avg_color[2]
            
            # Save the features and the true label you assigned
            X.append([avg_r, avg_g, avg_b])
            y.append(label_idx)
            
    if len(X) < 4:
        print("⚠️ Not enough training data yet. Collect at least 4 corrections in your app first!")
        return
        
    # Train a smart decision boundary classifier based strictly on your changes
    print(f"🧠 Training model on {len(X)} corrected samples...")
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save the smart model file to disk
    with open("hemolysis_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("✅ Success! 'hemolysis_model.pkl' created and updated based on your feedback.")

if __name__ == "__main__":
    train_new_model()
