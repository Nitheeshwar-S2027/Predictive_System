"""Streamlit frontend for Predictive_System.

This app accepts either a single comma-separated text input for all features
or a single microphone recording (client-side) which is transcribed and
parsed. It mirrors feature-engineering and preprocessor usage from
`predictor.py` to improve prediction consistency/accuracy and supports
PDF report generation + optional email sending.

Notes:
- Uses `input.py` helper functions (parse_number, initialize_voice_model, stt model).
- Falls back to typed input if server-side STT is not available.
"""

import os
import tempfile
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st

from models import HybridDNN, HybridCNN, HybridGRU
import input as input_mod

# Only import DATA_CONFIG, don't force report_utils imports indirectly
from predictor import DATA_CONFIG

# Import PDF functions on demand to avoid import cycles
REPORT_UTILS_IMPORTED = False
generate_pdf_report = None
send_real_email = None

try:
    # optional client-side recorder; not required but preferred
    from streamlit_audiorecorder import audiorecorder

    HAS_AUDIOTR = True
except Exception:
    HAS_AUDIOTR = False

# --- UI constants ---
STATUS_COLORS = {
    "redstatus": "#dc143c",
    "orangestatus": "#ff8c00",
    "greenstatus": "#228b22",
    "yellowstatus": "#ffd700",
}


def list_models_for(vehicle: str) -> List[str]:
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    prefix = f"{vehicle}_"
    files = [
        f
        for f in os.listdir(models_dir)
        if f.startswith(prefix) and f.endswith((".pt", ".pkl"))
    ]
    return sorted(files)


def save_audio_bytes_to_wav(audio_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def load_model(model_path: str, input_size: int):
    """Load either a PyTorch .pt hybrid model or a joblib .pkl model."""
    if model_path.endswith(".pt"):
        base = os.path.basename(model_path)
        if "hybrid_dnn" in base or "dnn" in base:
            m = HybridDNN(input_size)
        elif "cnn" in base:
            m = HybridCNN(input_size)
        elif "gru" in base:
            m = HybridGRU(input_size)
        else:
            raise RuntimeError(f"Unknown .pt model type: {base}")
        state = torch.load(model_path, map_location=torch.device("cpu"))
        m.load_state_dict(state)
        m.eval()
        return m, "deep"
    else:
        mdl = joblib.load(model_path)
        return mdl, "ml"


def predict_with_model(model, model_type: str, X_input: np.ndarray, task: str):
    if model_type == "deep":
        X_tensor = torch.from_numpy(X_input).float()
        with torch.no_grad():
            out = model(X_tensor)
        if task == "classification":
            prob = float(torch.sigmoid(out).numpy().flatten()[0])
            pred = int(prob > 0.5)
            return pred, prob
        else:
            val = float(out.numpy().flatten()[0])
            return val, None
    else:
        pred = model.predict(X_input)[0]
        prob = None
        if task == "classification" and hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_input)[0][1])
        elif task == "classification":
            try:
                prob = float(pred)
            except Exception:
                prob = None
        return pred, prob


def main():
    st.set_page_config(page_title="Predictive Maintenance", layout="wide")
    st.title("Predictive Maintenance - Web Interface")

    # Sidebar controls
    st.sidebar.header("Configuration")
    st.sidebar.markdown("### 1. Vehicle Selection")
    vehicle = st.sidebar.selectbox("Vehicle type", options=list(DATA_CONFIG.keys()))
    config = DATA_CONFIG[vehicle]
    model_files = list_models_for(vehicle)

    if not model_files:
        st.sidebar.error(
            "No trained models found in ./models. Run Model_Trainer to create models."
        )

    st.sidebar.markdown("### 2. Model Selection")
    model_file = st.sidebar.selectbox("Select model", options=model_files)

    # Audio settings in sidebar for better organization
    st.sidebar.markdown("### 3. Audio Settings")
    enable_stt = st.sidebar.checkbox(
        "Enable voice recognition",
        value=False,
        help="Uses faster-whisper for accurate speech recognition",
    )

    if enable_stt and not input_mod.STT_AVAILABLE:
        st.sidebar.warning(
            "⚠️ Voice recognition not available. Run setup_voice.py first"
        )

    if enable_stt and input_mod.STT_AVAILABLE and input_mod.stt_model is None:
        if st.sidebar.button("Initialize Voice Model"):
            with st.spinner("Loading voice model..."):
                input_mod.initialize_voice_model()
                st.sidebar.success("✅ Voice model ready")

    # Main content area
    st.markdown("## Input Mode")
    input_col1, input_col2 = st.columns([1, 1])
    with input_col1:
        input_mode = st.radio(
            "How would you like to provide input?",
            ["Text input", "Voice input"],
            help="Voice input requires microphone access",
        )

    # Show expected features to the user (step 2)
    st.markdown("## Step 2 — Features & Model")
    st.markdown("**Expected input features (in order):**")
    st.write(config["input_features"])

    # Show selected model info
    st.markdown("**Selected Model:**")
    st.info(
        model_file.replace(vehicle + "_", "") if model_file else "No model selected"
    )

    # Allow user to force using original numeric features (CLI had this behavior implicitly)
    force_use_original = st.checkbox(
        "Force use original numeric features (if available)"
    )

    # Sidebar: option to initialize server-side STT (faster_whisper)
    st.sidebar.markdown("---")
    enable_stt = st.sidebar.checkbox("Enable server STT (faster_whisper)", value=False)

    raw_inputs: Dict[str, object] = {}
    transcript = ""

    if input_mode.startswith("Text"):
        inp = st.text_input(
            "Enter all values, comma-separated (match the order shown above)"
        )
        if inp:
            parts = [p.strip() for p in inp.split(",")]
            if len(parts) != len(config["input_features"]):
                st.warning(
                    f'Expected {len(config["input_features"]) } values, got {len(parts)}'
                )
            else:
                for feat, val in zip(config["input_features"], parts):
                    if feat.lower() == "product id":
                        v = val.strip().upper()
                        if v and v[0] in ("L", "M", "H"):
                            raw_inputs[feat] = v[0]
                        else:
                            raw_inputs[feat] = v
                    else:
                        parsed = input_mod.parse_number(val)
                        raw_inputs[feat] = parsed if parsed is not None else val

    else:  # Voice input mode
        st.markdown(
            """
        ### Voice Input Instructions
        1. Speak values clearly, one at a time
        2. Pause briefly between values
        3. Say "high", "medium", or "low" for Product ID
        4. Numbers can be spoken naturally (e.g., "five point three" or "5.3")
        """
        )

        # Helper to parse transcript to tokens matching features
        def parse_transcript_to_tokens(transcript: str, features: List[str]):
            import re

            if not transcript:
                return []
            # try comma-separated first
            tokens = [
                t.strip()
                for t in transcript.replace(" and ", ",").split(",")
                if t.strip()
            ]
            if len(tokens) == 1:
                # whitespace fallback
                tokens = transcript.split()

            # if still mismatch, extract numeric substrings and try to map
            if len(tokens) != len(features):
                numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", transcript)
                tokens = []
                num_iter = iter(numbers)
                for feat in features:
                    if feat.lower() == "product id":
                        # detect quality words
                        found = None
                        for w in ("low", "medium", "high", "l", "m", "h"):
                            if w in transcript:
                                found = w
                                break
                        tokens.append(found if found else "")
                    else:
                        try:
                            tokens.append(next(num_iter))
                        except StopIteration:
                            tokens.append("")
            return tokens

        if HAS_AUDIOTR:
            if not enable_stt:
                st.warning("Please enable voice recognition in the sidebar first")
                return

            st.markdown("### 🎙️ Feature-by-Feature Recording")

            # Initialize session state for verification
            if "verified_values" not in st.session_state:
                st.session_state.verified_values = {}
            if "current_feature" not in st.session_state:
                st.session_state.current_feature = config["input_features"][0]
            if "current_value" not in st.session_state:
                st.session_state.current_value = None

            # Show progress
            total = len(config["input_features"])
            done = len(st.session_state.verified_values)
            st.progress(done / total)
            st.write(f"Recording {done + 1} of {total} features")

            # Display verified values so far
            if st.session_state.verified_values:
                st.markdown("### ✅ Verified Values")
                for feat, val in st.session_state.verified_values.items():
                    st.write(f"- {feat}: **{val}**")

            # Record current feature
            current_feature = st.session_state.current_feature
            st.markdown(f"### 🎤 Now Recording: {current_feature}")

            rec_col1, rec_col2 = st.columns([3, 1])
            with rec_col1:
                audio_bytes = audiorecorder("Start/Stop Recording", "Recording...")

                if audio_bytes:
                    try:
                        with rec_col2:
                            if st.button("🔄 Reset"):
                                st.session_state.current_value = None
                                st.rerun()

                        tmp_path = save_audio_bytes_to_wav(audio_bytes)
                        st.audio(tmp_path)

                        with st.spinner(f"Processing {current_feature}..."):
                            # Process speech for current feature
                            transcript = ""
                            if enable_stt and input_mod.STT_AVAILABLE:
                                if input_mod.stt_model is None:
                                    with st.status("Loading voice model..."):
                                        input_mod.initialize_voice_model()
                                try:
                                    segments, _ = input_mod.stt_model.transcribe(
                                        tmp_path
                                    )
                                    transcript = (
                                        " ".join(s.text for s in segments)
                                        .lower()
                                        .strip()
                                    )

                                    # Parse value based on feature type
                                    if current_feature.lower() == "product id":
                                        if "high" in transcript or "h" == transcript:
                                            value = "H"
                                        elif (
                                            "medium" in transcript or "m" == transcript
                                        ):
                                            value = "M"
                                        elif "low" in transcript or "l" == transcript:
                                            value = "L"
                                        else:
                                            value = None
                                    else:
                                        value = input_mod.parse_number(transcript)

                                    if value is not None:
                                        st.session_state.current_value = value
                                        # Show verification UI
                                        msg = f"Heard {value}, is this correct?"
                                        st.info(msg)
                                        input_mod.speak_local(msg)

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if st.button("✅ Yes"):
                                                st.session_state.verified_values[
                                                    current_feature
                                                ] = value
                                                feat_idx = config[
                                                    "input_features"
                                                ].index(current_feature)
                                                if feat_idx + 1 < len(
                                                    config["input_features"]
                                                ):
                                                    next_feat = config[
                                                        "input_features"
                                                    ][feat_idx + 1]
                                                    st.session_state.current_feature = (
                                                        next_feat
                                                    )
                                                    st.session_state.current_value = (
                                                        None
                                                    )
                                                    input_mod.speak_local(
                                                        "Moving to next"
                                                    )
                                                else:
                                                    input_mod.speak_local("All done")
                                                st.rerun()
                                        with col2:
                                            if st.button("❌ No"):
                                                st.session_state.current_value = None
                                                input_mod.speak_local("Try again")
                                                st.rerun()
                                    else:
                                        st.error("Could not understand. Try again.")
                                        input_mod.speak_local("Try again")
                                except Exception as e:
                                    st.error(f"Recognition error: {e}")
                            else:
                                st.info("Enable voice recognition first")
                        # Show transcription for debugging (if any)
                        if transcript:
                            st.info(f"Heard: {transcript}")
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
        else:
            uploaded = st.file_uploader(
                "Upload a WAV audio file (fallback)", type=["wav", "mp3", "m4a"]
            )
            if uploaded is not None:
                try:
                    b = uploaded.read()
                    tmp_path = save_audio_bytes_to_wav(b)
                    st.audio(tmp_path)
                    transcript = ""
                    if enable_stt and input_mod.STT_AVAILABLE:
                        if input_mod.stt_model is None:
                            input_mod.initialize_voice_model()
                        try:
                            segments, _ = input_mod.stt_model.transcribe(tmp_path)
                            transcript = (
                                " ".join([s.text for s in segments]).lower().strip()
                            )
                            st.info(f"Transcript: {transcript}")
                        except Exception as e:
                            st.error(f"Audio transcription failed: {e}")
                    else:
                        st.info("Server STT not enabled or available.")

                    tokens = parse_transcript_to_tokens(
                        transcript, config["input_features"]
                    )
                    if len(tokens) == len(config["input_features"]):
                        for feat, tok in zip(config["input_features"], tokens):
                            if feat.lower() == "product id":
                                tok_u = (tok or "").strip().upper()
                                raw_inputs[feat] = (
                                    tok_u[0]
                                    if tok_u and tok_u[0] in ("L", "M", "H")
                                    else tok_u
                                )
                            else:
                                parsed = input_mod.parse_number(tok)
                                raw_inputs[feat] = (
                                    parsed
                                    if parsed is not None
                                    else (tok if tok else None)
                                )
                    else:
                        st.warning(
                            "Could not parse expected number of values from audio"
                        )
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    # If we have all verified inputs, allow prediction
    if input_mode.startswith("Voice") and "verified_values" in st.session_state:
        if len(st.session_state.verified_values) == len(config["input_features"]):
            raw_inputs = st.session_state.verified_values
            st.markdown("### ✅ All Values Verified")
            st.json(raw_inputs)
            input_mod.speak_local(
                "All values have been verified. Ready for prediction."
            )

    # For text input mode or if we have raw inputs
    if raw_inputs:
        if not input_mode.startswith("Voice"):
            st.write("Captured inputs:")
            st.json(raw_inputs)

        if st.button("Run Prediction"):
            # Build DataFrame matching predictor's expectations
            try:
                X_raw = pd.DataFrame([raw_inputs])
                # Ensure numeric conversions for non-Product ID columns
                for c in list(X_raw.columns):
                    if c.lower() != "product id":
                        try:
                            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
                        except Exception:
                            pass

                # Feature engineering for engine
                if vehicle == "engine":
                    X_raw["Temp_Diff"] = (
                        X_raw["Process temperature [K]"] - X_raw["Air temperature [K]"]
                    )
                    X_raw["Power_Proxy"] = (
                        X_raw["Torque [Nm]"] * X_raw["Rotational speed [rpm]"]
                    )
                    X_raw["Overstrain_Proxy"] = (
                        X_raw["Tool wear [min]"] * X_raw["Torque [Nm]"]
                    )
                    X_final = X_raw[
                        config["final_numerical_features"]
                        + config["categorical_features"]
                    ]
                else:
                    # EV uses final numerical as-is
                    X_final = X_raw[config["final_numerical_features"]]

                # Load preprocessor if available and show diagnostics
                preproc_path = os.path.join("models", f"{vehicle}_preprocessor.pkl")
                if os.path.exists(preproc_path):
                    preprocessor = joblib.load(preproc_path)
                    # Diagnostics: try to show expected feature names / ordering
                    try:
                        names_in = getattr(preprocessor, "feature_names_in_", None)
                        if names_in is not None:
                            st.info(
                                f"Preprocessor expects these input columns (n={len(names_in)}): {list(names_in)[:20]}"
                            )
                            # attempt to reorder X_final columns to match preprocessor if possible
                            cols_lower = {c.lower(): c for c in X_final.columns}
                            reorder = []
                            missing = []
                            for req in names_in:
                                key = req.lower()
                                if key in cols_lower:
                                    reorder.append(cols_lower[key])
                                else:
                                    missing.append(req)
                            if missing:
                                st.warning(
                                    f"Preprocessor expected columns not found in input (sample): {missing[:5]}"
                                )
                            if reorder:
                                try:
                                    X_to_transform = X_final[reorder]
                                except Exception:
                                    X_to_transform = X_final
                            else:
                                X_to_transform = X_final
                        else:
                            X_to_transform = X_final
                    except Exception:
                        X_to_transform = X_final

                    X_scaled = preprocessor.transform(X_to_transform)
                    if not isinstance(X_scaled, np.ndarray):
                        try:
                            X_scaled = X_scaled.toarray()
                        except Exception:
                            X_scaled = np.asarray(X_scaled)
                else:
                    st.info("Preprocessor not found; using raw values")
                    X_scaled = X_final.values

                X_input = np.asarray(X_scaled).reshape(1, -1).astype(np.float32)

                # Load selected model
                model_path = os.path.join("models", model_file)
                model_obj, model_type = load_model(model_path, X_input.shape[1])

                # Detect input-shape mismatch for ML models and attempt fallbacks
                if model_type == "ml":
                    expected = getattr(model_obj, "n_features_in_", None)
                    actual = X_input.shape[1]
                    if expected is not None and expected != actual:
                        st.warning(
                            f"Model expects {expected} features but input has {actual}. Trying fallbacks..."
                        )
                        fallback_done = False

                        # If user explicitly requests original numeric features, try that first
                        if force_use_original:
                            orig_feats = DATA_CONFIG[vehicle].get(
                                "original_numerical_features"
                            )
                            if orig_feats and all(
                                f in X_raw.columns for f in orig_feats
                            ):
                                try:
                                    X_orig = X_raw[orig_feats].astype(float)
                                    X_input = X_orig.values.reshape(1, -1).astype(
                                        np.float32
                                    )
                                    if X_input.shape[1] == expected:
                                        st.info(
                                            "Using original numerical features as forced fallback to match model input."
                                        )
                                        fallback_done = True
                                except Exception as e:
                                    st.info(
                                        f"Could not use original numerical features: {e}"
                                    )

                        # 1) If we have a preprocessor, check its transform shape (only if not forced)
                        if not fallback_done and "preprocessor" in locals():
                            try:
                                test = preprocessor.transform(X_to_transform)
                                if not isinstance(test, np.ndarray):
                                    try:
                                        test = test.toarray()
                                    except Exception:
                                        test = np.asarray(test)
                                if test.reshape(1, -1).shape[1] == expected:
                                    X_input = (
                                        np.asarray(test)
                                        .reshape(1, -1)
                                        .astype(np.float32)
                                    )
                                    st.info(
                                        "Adjusted input using preprocessor output to match model expected dims."
                                    )
                                    fallback_done = True
                            except Exception as e:
                                st.info(f"Preprocessor transform check failed: {e}")

                        # 2) Try original numerical features (useful if model was trained on raw features)
                        if not fallback_done:
                            orig_feats = DATA_CONFIG[vehicle].get(
                                "original_numerical_features"
                            )
                            if orig_feats and all(
                                f in X_raw.columns for f in orig_feats
                            ):
                                try:
                                    X_orig = X_raw[orig_feats].astype(float)
                                    X_input = X_orig.values.reshape(1, -1).astype(
                                        np.float32
                                    )
                                    if X_input.shape[1] == expected:
                                        st.info(
                                            "Using original numerical features as fallback to match model input."
                                        )
                                        fallback_done = True
                                except Exception as e:
                                    st.info(
                                        f"Could not use original numerical features: {e}"
                                    )

                        if not fallback_done:
                            st.error(
                                "Could not automatically reconcile model input dimensions. Please check preprocessor and model consistency."
                            )

                pred, prob = predict_with_model(
                    model_obj, model_type, X_input, config["task"]
                )

                # Build display
                if config["task"] == "classification":
                    status = (
                        "HIGH RISK OF FAILURE" if int(pred) == 1 else "NORMAL OPERATION"
                    )
                    confidence = f"{prob:.1%}" if prob is not None else "N/A"
                    color = "redstatus" if int(pred) == 1 else "greenstatus"

                    # Visual display
                    st.markdown(f"**Prediction:** {status}")
                    st.markdown(f"**Confidence:** {confidence}")

                    # Audio narration
                    narration = f"Analysis complete. {status} detected"
                    if prob is not None:
                        narration += f" with {prob:.0%} confidence"
                    input_mod.speak_local(narration)
                else:
                    rul = int(round(float(pred), 0))
                    # Visual display
                    st.markdown(f"**Estimated Remaining Useful Life:** {rul} cycles")

                    # Status color and narration
                    if rul < 50:
                        color = "redstatus"
                        msg = f"Warning: Only {rul} cycles remaining. Immediate maintenance required."
                    elif rul < 200:
                        color = "orangestatus"
                        msg = f"Caution: {rul} cycles remaining. Plan maintenance soon."
                    else:
                        color = "greenstatus"
                        msg = f"Good condition. {rul} cycles remaining until next maintenance."

                    # Audio narration
                    input_mod.speak_local(msg)

                st.markdown(
                    f'<div style="padding:12px;border-radius:6px;background-color:{STATUS_COLORS[color]};color:white">'
                    f"<b>Summary:</b> Prediction completed for {vehicle.upper()}</div>",
                    unsafe_allow_html=True,
                )

                # Simple mock explanation (replaceable with SHAP later)
                st.markdown("### Explanation (approx)")
                imp = np.abs(np.random.rand(X_input.shape[1]))
                top_idx = np.argsort(imp)[::-1][:3]
                feat_names = (
                    config["final_numerical_features"] + config["categorical_features"]
                )
                for i in top_idx:
                    name = feat_names[i] if i < len(feat_names) else f"Feature_{i}"
                    st.write(f"- {name}: importance {imp[i]:.3f}")

                # Audio summary generation (server-side TTS to WAV)
                if st.button("Play Audio Summary"):
                    summary = f"Prediction result for {vehicle}: {status if config['task']=='classification' else 'RUL '+str(rul)}"
                    # Prefer using the input module's TTS wrapper which is optional and safe
                    try:
                        if getattr(input_mod, "TTS_AVAILABLE", False):
                            input_mod.speak_local(summary)
                            st.success("Spoken summary played using local TTS.")
                        else:
                            st.warning(
                                "Text-to-speech (pyttsx3) is not available in this environment.\n"
                                "Install it (pip install pyttsx3) or enable TTS to play spoken summaries."
                            )
                    except Exception as e:
                        st.error(f"Could not play audio summary: {e}")

                # PDF + Email
                if st.checkbox("Generate PDF report"):
                    # Import report functions on demand
                    global REPORT_UTILS_IMPORTED, generate_pdf_report, send_real_email
                    if not REPORT_UTILS_IMPORTED:
                        try:
                            from report_utils import (
                                generate_pdf_report,
                                send_real_email,
                            )

                            REPORT_UTILS_IMPORTED = True
                        except ImportError as e:
                            st.error(f"Could not import PDF/email functions: {e}")
                            st.info("Install required packages: pip install fpdf2")
                            return

                    recipient = st.text_input(
                        "Recipient email (leave blank to skip sending)"
                    )
                    if st.button("Create & (optionally) send report"):
                        reports_dir = "reports"
                        os.makedirs(reports_dir, exist_ok=True)
                        fname = f"report_{vehicle}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        outpath = os.path.join(reports_dir, fname)
                        explanation_summary = "Auto-generated explanation (approx)."
                        top_feature = (
                            feat_names[top_idx[0]] if top_idx.size else feat_names[0]
                        )

                        try:
                            generate_pdf_report(
                                raw_inputs,
                                vehicle,
                                status,
                                color,
                                (rul if config["task"] != "classification" else 0.0),
                                explanation_summary,
                                top_feature,
                                outpath,
                            )
                            st.success(f"PDF saved to {outpath}")

                            if recipient and "@" in recipient:
                                sent = send_real_email(
                                    recipient,
                                    f"PM Report - {vehicle.upper()}",
                                    "See attached report",
                                    outpath,
                                )
                                if sent:
                                    st.success("Email sent")
                                else:
                                    st.warning(
                                        "Email not sent (check SMTP configuration)"
                                    )
                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")
                            if "fpdf2" in str(e):
                                st.info(
                                    "Install the PDF package with: pip install fpdf2"
                                )

            except Exception as e:
                st.error(f"Prediction pipeline failed: {e}")


if __name__ == "__main__":
    main()
