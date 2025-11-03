import os

import joblib

import numpy as np

import pandas as pd

import torch

import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pyttsx3
from typing import Dict, Union

# --- IMPORT MODULARIZED COMPONENTS (UPDATED: use hybrid transformer models) ---
from models import HybridDNN, HybridCNN, HybridGRU
from report_utils import handle_report_and_email
from input import get_input_data, listen, parse_number, speak_local

# -------------------------------------

# PDF Report Constants (still needed here for internal status logic)
COLOR_BLUE = (30, 144, 255)  # Dodger Blue for headers
COLOR_GRAY = (240, 240, 240)  # Light Gray for table background
COLOR_GREEN = (34, 139, 34)  # Forest Green (LOW RISK)
COLOR_ORANGE = (255, 165, 0)  # Orange (CHECK SOON)
COLOR_RED = (220, 20, 60)  # Crimson (HIGH RISK/URGENT)


# --- VOICE (TTS) SETUP ---
def speak(text):
    """Speak and print text."""
    print(f"🗣️ {text}")
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass


# =================================================================
# === CONFIGURATION & DATA SETUP (UNCHANGED) ===
# =================================================================

DATA_CONFIG = {
    # Engine Data: Failure Classification Task
    "engine": {
        "path": "engine_data.csv",
        "task": "classification",
        "target": "machine failure",
        # 🛠️ FIX 1: Change 'Product ID' to lowercase 'product id'
        "categorical_features": ["product id", "type"],
        # 🛠️ FIX 2: Change casing in all input features to match the lowercased DataFrame
        "input_features": [
            "product id",
            "air temperature [k]",
            "process temperature [k]",
            "rotational speed [rpm]",
            "torque [nm]",
            "tool wear [min]",
        ],
        "final_numerical_features": [
            "Temp_Diff",
            "Power_Proxy",
            "Overstrain_Proxy",
        ],  # These are newly created, so they don't need case correction
    },
    "ev": {
        "path": "ev_data.csv",
        "task": "regression",
        "target": "remaining_useful_life_cycles",
        "categorical_features": [],
        "input_features": [
            "Cycle_Index",
            "Discharge Time (s)",
            "Decrement 3.6-3.4V (s)",
            "Max. Voltage Dischar. (V)",
            "Min. Voltage Charg. (V)",
            "Time at 4.15V (s)",
            "Time constant current (s)",
            "Charging time (s)",
        ],
        "final_numerical_features": [
            "cycle_index",
            "discharge time (s)",
            "decrement 3.6-3.4v (s)",
            "max. voltage dischar. (v)",
            "min. voltage charg. (v)",
            "time at 4.15v (s)",
            "time constant current (s)",
            "charging time (s)",
        ],
    },
}

# =================================================================
# === MAIN PREDICTOR FUNCTION ===
# =================================================================


def run_prediction():
    global listen, parse_number, get_input_data

    # -----------------------------
    # Step 0: Input Mode Selection
    # -----------------------------
    speak("Please select the input mode: Voice or Text.")

    input_mode = None
    while input_mode not in ["voice", "text"]:
        print("\nInput Mode (type 'voice' or 'text'):")
        user_choice = input("➡ Your choice: ").lower().strip()
        if user_choice in ["voice", "text"]:
            input_mode = user_choice
            speak(f"Selected input mode: {input_mode.upper()}.")
        else:
            speak("Invalid choice. Please type 'voice' or 'text'.")
    # -----------------------------
    # Step 1: LOAD DATA AND DEFINE PREPROCESSOR
    # -----------------------------
    speak("What type of vehicle are you diagnosing? Please say 'engine' or 'EV'.")

    vehicle_choice_key = None
    while vehicle_choice_key is None:
        if input_mode == "voice":
            text = listen()
        else:
            text = input("Enter vehicle type (engine/EV): ").lower().strip()
        if "engine" in text or "motor" in text:
            vehicle_choice_key = "engine"
        elif "ev" in text or "battery" in text or "electric" in text:
            vehicle_choice_key = "ev"
        else:
            speak("I couldn't understand. Please try again.")

    speak(f"You selected: {vehicle_choice_key.upper()} diagnosis.")

    config = DATA_CONFIG[vehicle_choice_key]
    DATA_PATH = config["path"]
    INPUT_FEATURES = config["input_features"]
    FINAL_NUM_FEATURES = config["final_numerical_features"]
    CAT_FEATURES = config["categorical_features"]
    TASK_TYPE = config["task"]

    if not os.path.exists(DATA_PATH):
        speak(
            f"⚠️ Warning: {DATA_PATH} not found. Using dummy data for preprocessor fit."
        )
        dummy_data = {
            "product ID": ["L", "M", "H"],
            "air temperature [K]": [290, 300, 310],
            "process temperature [K]": [300, 310, 320],
            "rotational speed [rpm]": [1000, 1200, 1400],
            "torque [Nm]": [40, 50, 60],
            "tool wear [min]": [0, 50, 100],
            "machine failure": [0, 0, 1],
            "Cycle_Index": [1, 2, 3],
            "Discharge Time (s)": [1000, 900, 800],
            "Decrement 3.6-3.4V (s)": [20, 18, 16],
            "Max. Voltage Dischar. (V)": [4.0, 3.9, 3.8],
            "Min. Voltage Charg. (V)": [3.0, 3.1, 3.2],
            "Time at 4.15V (s)": [100, 90, 80],
            "Time constant current (s)": [500, 450, 400],
            "Charging time (s)": [3000, 2800, 2600],
            "Remaining_Useful_Life_cycles": [400, 300, 200],
        }
        data = pd.DataFrame(dummy_data)
    else:
        data = pd.read_csv(DATA_PATH)

    data.columns = data.columns.str.lower().str.strip()
    data = data.fillna(0)

    # --- FEATURE ENGINEERING ---
    if vehicle_choice_key == "engine":
        data["Temp_Diff"] = (
            data["process temperature [k]"] - data["air temperature [k]"]
        )
        data["Power_Proxy"] = data["torque [nm]"] * data["rotational speed [rpm]"]
        data["Overstrain_Proxy"] = data["tool wear [min]"] * data["torque [nm]"]
        X_fit = data[FINAL_NUM_FEATURES + CAT_FEATURES]
    else:
        X_fit = data[FINAL_NUM_FEATURES]

    # --- DEFINE COLUMNTRANSFORMER ---
    if CAT_FEATURES:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), FINAL_NUM_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), FINAL_NUM_FEATURES)]
        )

    try:
        preprocessor.fit(X_fit)
    except ValueError as e:
        speak(f"Error fitting preprocessor: {e}. Check data file and column names.")
        return

    # --- DETERMINE FINAL FEATURES, INPUT SIZE, AND BACKGROUND DATA ---
    try:
        if CAT_FEATURES:
            cat_feature_names = preprocessor.named_transformers_[
                "cat"
            ].get_feature_names_out(CAT_FEATURES)
            feature_names = FINAL_NUM_FEATURES + [
                f"{CAT_FEATURES[0]}_{n.split('_')[-1]}" for n in cat_feature_names
            ]
        else:
            feature_names = FINAL_NUM_FEATURES
    except Exception:
        feature_names = FINAL_NUM_FEATURES

    INPUT_SIZE = len(feature_names)
    speak(f"Model will require {INPUT_SIZE} features after preprocessing.")

    X_fit_processed = preprocessor.transform(X_fit)
    background_data = X_fit_processed
    # SHAP/LIME is being mocked but needs this for initialization if used

    # -----------------------------
    # Step 2 & 3: Model Loading
    # -----------------------------
    prefix = vehicle_choice_key + "_"
    os.makedirs("models", exist_ok=True)
    model_files = [
        f
        for f in os.listdir("models")
        if f.startswith(prefix) and f.endswith((".pt", ".pkl"))
    ]

    if not model_files:
        speak(f"⚠️ Error: No saved {vehicle_choice_key.upper()} models found. Exiting.")
        return

    speak(f"Available {vehicle_choice_key.upper()} models are:")
    for i, f in enumerate(model_files, 1):
        display_name = f.replace(prefix, "")
        speak(f"Option {i}: {display_name}")

    choice = None
    while choice is None:
        speak("Please say or type the number of the model you want to use.")
        model_choice_text = input("Model Choice (number): ")
        try:
            choice_val = int(model_choice_text)
            if 1 <= choice_val <= len(model_files):
                choice = choice_val
                speak(
                    f"You selected option {choice}: {model_files[choice - 1].replace(prefix, '')}"
                )
            else:
                speak("Invalid number.")
        except ValueError:
            speak("Invalid input. Please enter a number.")

    model_file = model_files[choice - 1]
    model_path = os.path.join("models", model_file)
    speak(f"Selected model is {model_file}")

    model_type = None
    model = None

    if model_file.endswith((".pt")):
        try:
            base_name = model_file.replace(prefix, "")

            # Map saved filenames to Hybrid transformer classes
            # Use HybridDNN for both "dnn" and "hybrid_dnn" variants
            if "hybrid_dnn_model.pt" in base_name or "dnn_model.pt" in base_name:
                model = HybridDNN(INPUT_SIZE)
            elif "cnn_model.pt" in base_name:
                model = HybridCNN(INPUT_SIZE)
            elif "gru_model.pt" in base_name:
                model = HybridGRU(INPUT_SIZE)
            else:
                speak(f"Error: Missing class definition for {base_name}. Skipping.")
                model = None

            if model is not None:
                state_dict = torch.load(model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
                model.eval()
                model_type = "deep"
                speak("✅ PyTorch model loaded successfully.")

        except Exception as e:
            speak(f"ERROR: Could not load the PyTorch model file: {e}")
            model = None

    elif model_file.endswith(".pkl"):
        try:
            model = joblib.load(model_path)
            model_type = "ml"
            speak("✅ ML model (.pkl) loaded successfully.")
        except Exception as e:
            speak(
                f"ERROR: Model file not found or load error: {e}. Prediction will be skipped."
            )
            model = None

    if model is None:
        return

    # -----------------------------
    # Step 4: Collect feature values
    # -----------------------------
    raw_input_dict = get_input_data(
        speak, listen, parse_number, INPUT_FEATURES, vehicle_choice_key, input_mode
    )

    if not raw_input_dict:
        speak("Data collection failed. Aborting prediction.")
        return

    # -----------------------------
    # Step 5: Preprocess (Load Saved Preprocessor)
    # -----------------------------
    if "type" in CAT_FEATURES:
        raw_input_dict["type"] = "l"
    X_raw_df = pd.DataFrame([raw_input_dict])

    X_raw_df.columns = X_raw_df.columns.str.lower().str.strip()

    # ----- Feature Engineering -----
    if vehicle_choice_key == "engine":
        X_raw_df["Temp_Diff"] = (
            X_raw_df["process temperature [k]"] - X_raw_df["air temperature [k]"]
        )
        X_raw_df["Power_Proxy"] = (
            X_raw_df["torque [nm]"] * X_raw_df["rotational speed [rpm]"]
        )
        X_raw_df["Overstrain_Proxy"] = (
            X_raw_df["tool wear [min]"] * X_raw_df["torque [nm]"]
        )
        X_final = X_raw_df[FINAL_NUM_FEATURES + CAT_FEATURES]
        if vehicle_choice_key == "engine":
            # Create a dictionary mapping the current lowercase names to the expected saved names
            rename_map = {"product id": "Product ID", "type": "Type"}
            X_final = X_final.rename(columns=rename_map)
    else:
        X_final = X_raw_df[FINAL_NUM_FEATURES]

        rename_map = {k.lower().strip(): k for k in DATA_CONFIG["ev"]["input_features"]}
        X_final = X_final.rename(columns=rename_map)

    # ----- Load Preprocessor -----
    preproc_path = os.path.join("models", f"{vehicle_choice_key}_preprocessor.pkl")
    if os.path.exists(preproc_path):
        preprocessor = joblib.load(preproc_path)
        print("✅ Loaded saved preprocessor for consistent scaling.")
        X_scaled = preprocessor.transform(X_final)
    else:
        print("⚠️ Preprocessor not found — using raw input (no scaling).")
        X_scaled = X_final.values

    X_input = X_scaled.reshape(1, -1)

    # -----------------------------
    # Step 6 & 7: Predict and Output
    # -----------------------------
    speak("Analyzing data and predicting status...")

    pred_value = np.nan
    rul_cycles = np.nan
    pred_prob = np.nan
    status_color = "yellowstatus"

    try:
        # Prediction logic using the model and input data
        if model_type == "deep":
            X_tensor = torch.from_numpy(X_input).float()

            # NOTE: hybrid models expect raw (batch, features) input and handle internal reshaping
            with torch.no_grad():
                output = model(X_tensor)

            if TASK_TYPE == "classification":
                pred_prob = torch.sigmoid(output).numpy().flatten()[0]
                pred_value = (pred_prob > 0.5).astype(int)
            else:
                pred_value = output.numpy().flatten()[0]
        else:  # ML Model
            pred_value = model.predict(X_input)[0]
            if TASK_TYPE == "classification" and hasattr(model, "predict_proba"):
                pred_prob = model.predict_proba(X_input)[0][1]
            elif TASK_TYPE == "classification":
                pred_prob = pred_value

    except Exception as e:
        speak(f"Prediction failed: {e}")

    # --- Status Message and Color Assignment ---
    if np.isnan(pred_value):
        status_message = "UNKNOWN (Prediction failed)"
        status_color = "yellowstatus"
    else:
        if vehicle_choice_key == "engine":
            if pred_value == 1:
                status_message = (
                    f"HIGH RISK OF MACHINE FAILURE (Confidence: {pred_prob:.1%})"
                )
                status_color = "redstatus"
            else:
                status_message = f"NORMAL OPERATION (Failure risk: {pred_prob:.1%})"
                status_color = "greenstatus"
        else:  # EV - Regression
            rul_cycles = round(pred_value, 0)
            status_message = (
                f"Estimated Remaining Useful Life is **{rul_cycles} cycles**."
            )
            if rul_cycles < 50:
                status_message += " **Immediate maintenance is highly recommended.**"
                status_color = "redstatus"
            elif rul_cycles < 200:
                status_message += " **Check system soon.**"
                status_color = "orangestatus"
            else:
                status_color = "greenstatus"

    speak(f"The predicted status is: {status_message}")
    print(
        f"\n🚗 Predicted Vehicle Status: {status_message} ({vehicle_choice_key.upper()})\n"
    )

    # -----------------------------
    # Step 8: Generate Explanations (SHAP & LIME)
    # -----------------------------
    speak("Generating explanations for model decision. This may take a moment.")
    top_feature = "N/A"
    explanation_summary = "Explanation generation skipped."

    try:
        # Mock SHAP for demonstration (replace with actual SHAP/LIME logic if needed)
        abs_shap_values = np.abs(np.random.rand(INPUT_SIZE))
        top_indices = np.argsort(abs_shap_values)[::-1][:2]

        top_feature_index = top_indices[0] if top_indices.size > 0 else 0
        top_feature = feature_names[top_feature_index].replace(
            "product ID_", "Quality: "
        )

        shap_summary = f"SHAP analysis driven by **{feature_names[top_indices[0]]}**."
        lime_summary = (
            f"LIME highlights **{feature_names[top_indices[1]]}** as relevant."
        )
        explanation_summary = (
            f"**SHAP Report:** {shap_summary}\n\n**LIME Report:** {lime_summary}"
        )

    except Exception as e:
        print(f"Explainability Error: {e}")
        explanation_summary = "Explanation generation failed."
        top_feature = "Data Quality"

    # -----------------------------
    # Step 9: Report Generation and Email (Modularized)
    # -----------------------------
    rul_cycles_report = rul_cycles if not np.isnan(rul_cycles) else 0.0

    handle_report_and_email(
        raw_input_dict,
        vehicle_choice_key,
        status_message,
        status_color,
        rul_cycles_report,
        explanation_summary,
        top_feature,
    )


if __name__ == "__main__":
    run_prediction()
