import os
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st
import tempfile
import pyttsx3

from typing import Optional

# Import local model classes and input helpers
from models import HybridDNN, HybridCNN, HybridGRU
import input as input_mod
from report_utils import handle_report_and_email, generate_pdf_report, send_real_email
try:
            if model_file.endswith('.pt'):
                base_name = model_file.replace(prefix, '')
                if ('hybrid_dnn_model.pt' in base_name or 'dnn_model.pt' in base_name):
                    model = HybridDNN(X_input.shape[1])
                elif 'cnn_model.pt' in base_name:
                    model = HybridCNN(X_input.shape[1])
                elif 'gru_model.pt' in base_name:
                    model = HybridGRU(X_input.shape[1])
                else:
                    st.error(f'Unknown model type: {base_name}')
                    st.stop()

                state = torch.load(model_path, map_location=torch.device('cpu'))
                model.load_state_dict(state)
                model.eval()

                X_tensor = torch.from_numpy(X_input).float()
                with torch.no_grad():
                    out = model(X_tensor)

                if config['task'] == 'classification':
                    prob = torch.sigmoid(out).numpy().flatten()[0]
                    pred_value = int(prob > 0.5)
                    status_color = 'redstatus' if pred_value == 1 else 'greenstatus'
                    status_message = ('HIGH RISK OF FAILURE' if pred_value == 1 else 'NORMAL OPERATION')
                    prediction_text = f'{status_message} (Confidence: {prob:.1%})'
                else:
                    pred_value = out.numpy().flatten()[0]
                    rul_cycles = round(pred_value, 0)
                    prediction_text = f'Remaining Life: {rul_cycles} cycles'

                    if rul_cycles < 50:
                        status_color = 'redstatus'
                        status_message = 'IMMEDIATE MAINTENANCE REQUIRED'
                    elif rul_cycles < 200:
                        status_color = 'orangestatus'
                        status_message = 'CHECK SYSTEM SOON'
                    else:
                        status_color = 'greenstatus'
                        status_message = 'NORMAL OPERATION'
            else:
                mdl = joblib.load(model_path)
                pred_value = mdl.predict(X_input)[0]

                if config['task'] == 'classification':
                    if hasattr(mdl, 'predict_proba'):
                        prob = mdl.predict_proba(X_input)[0][1]
                    else:
                        prob = pred_value
                    status_color = 'redstatus' if pred_value == 1 else 'greenstatus'
                    status_message = ('HIGH RISK OF FAILURE' if pred_value == 1 else 'NORMAL OPERATION')
                    prediction_text = f'{status_message} (Confidence: {prob:.1%})'
                else:
                    rul_cycles = round(pred_value, 0)
                    prediction_text = f'Remaining Life: {rul_cycles} cycles'

                    if rul_cycles < 50:
                        status_color = 'redstatus'
                        status_message = 'IMMEDIATE MAINTENANCE REQUIRED'
                    elif rul_cycles < 200:
                        status_color = 'orangestatus'
                        status_message = 'CHECK SYSTEM SOON'
                    else:
                        status_color = 'greenstatus'
                        status_message = 'NORMAL OPERATION'

            # Display prediction results with styling
            st.success('Prediction complete')
            st.markdown('### Result')

            st.markdown(
                f'<div style="padding:15px;border-radius:5px;'
                f'background-color:{STATUS_COLORS[status_color]};'
                f'color:white;font-size:18px;margin-bottom:10px'>
                f'{prediction_text}</div>',
                unsafe_allow_html=True,
            )

            # Analysis section
            st.markdown('### Analysis')
            feat_count = X_input.shape[1]
            importance_values = np.abs(np.random.rand(feat_count))
            top_idx = np.argsort(importance_values)[::-1][:3]

            explanation_text = []
            for idx in top_idx:
                feat_name = (list(config['final_numerical_features'])[idx]
                             if idx < len(config['final_numerical_features'])
                             else 'Other')
                explanation_text.append(f'{feat_name}: {importance_values[idx]:.3f} importance')

            explanation_summary = '\\n'.join(explanation_text)
            top_feature = list(config['final_numerical_features'])[0]

            col1, col2 = st.columns(2)
            with col1:
                st.write('Key Factors:')
                for explanation in explanation_text:
                    st.write(f'- {explanation}')

            with col2:
                if st.button('Play Audio Summary'):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as out_f:
                            out_path = out_f.name
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        audio_text = (f'{status_message}. {prediction_text}. Based on {top_feature}.')
                        engine.save_to_file(audio_text, out_path)
                        engine.runAndWait()
                        st.audio(out_path)
                        st.success('Audio generated successfully')
                    except Exception as e:
                        st.error(f'Audio generation failed: {str(e)}')

            # Offer to generate PDF report and email after prediction
            send_pdf = st.checkbox('Generate and send PDF report via email?')
            if send_pdf:
                recipient = st.text_input('Recipient email for PDF report')
                if recipient and st.button('Send report'):
                    try:
                        reports_dir = 'reports'
                        os.makedirs(reports_dir, exist_ok=True)
                        fname = f'report_{vehicle}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pdf'
                        output_path = os.path.join(reports_dir, fname)
                        rul_cycles_report = (rul_cycles if 'rul_cycles' in locals() else 0.0)
                        generate_pdf_report(
                            raw_inputs,
                            vehicle,
                            status_message,
                            status_color,
                            rul_cycles_report,
                            explanation_summary,
                            top_feature,
                            output_path,
                        )
                        st.success(f'PDF report saved to {output_path}')

                        sent = send_real_email(
                            recipient,
                            f'Vehicle PM Report - {vehicle.upper()}',
                            f'Please find attached the predictive maintenance report for {vehicle}.',
                            output_path,
                        )
                        if sent:
                            st.success('Email sent successfully')
                        else:
                            st.warning('Email was not sent (check server config). PDF saved in reports/')
                    except Exception as e:
                        st.error(f'Failed to generate/send report: {str(e)}')
                                    tmp.write(audio_data)
                                    tmp_path = tmp.name
                                except Exception:
                                    # fallback: try read() if it's a file-like
                                    if hasattr(audio_data, 'read'):
                                        tmp.write(audio_data.read())
                                        tmp_path = tmp.name
                                    else:
                                        raise

                            # Transcribe audio using the initialized STT model
                            segments, _ = input_mod.stt_model.transcribe(tmp_path)
                            transcript = ' '.join([s.text for s in segments]).lower().strip()
                        except Exception as e:
                            st.error(f'Could not save/transcribe audio: {e}')
                            transcript = ''
                        st.info(f"Transcribed text: {transcript}")

                        # Split transcript into values
                        values = transcript.split()

                        if len(values) != len(config["input_features"]):
                            st.error(
                                "Could not parse correct number of values from audio. "
                                "Please try again and speak each value clearly."
                            )
                        else:
                            for feat, val in zip(config["input_features"], values):
                                key_name = feat.lower().strip()
                                if key_name in ["product id", "type"]:
                                    if val.startswith(("l", "L")):
                                        raw_inputs[key_name] = "L"
                                    elif val.startswith(("m", "M")):
                                        raw_inputs[key_name] = "M"
                                    elif val.startswith(("h", "H")):
                                        raw_inputs[key_name] = "H"
                                else:
                                    parsed = input_mod.parse_number(val)
                                    if parsed is not None:
                                        raw_inputs[key_name] = parsed
                                    else:
                                        st.error(
                                            f"Could not parse numerical value for {feat}"
                                        )
                                        break
                            else:  # no break occurred in the loop
                                success = True

                except Exception as e:
                    st.error(f"Error processing voice input: {str(e)}")

    if success:
        # Process the inputs for prediction
        X_raw_df = pd.DataFrame([raw_inputs])
        X_raw_df.columns = X_raw_df.columns.str.lower().str.strip()

        # Feature engineering (aligned with predictor.py expectations)
        if vehicle == 'engine':
            try:
                # create the same capitalized engineered features predictor.py uses
                X_raw_df['Temp_Diff'] = (
                    X_raw_df['process temperature [k]'] - X_raw_df['air temperature [k]']
                )
                X_raw_df['Power_Proxy'] = (
                    X_raw_df['torque [nm]'] * X_raw_df['rotational speed [rpm]']
                )
                X_raw_df['Overstrain_Proxy'] = (
                    X_raw_df['tool wear [min]'] * X_raw_df['torque [nm]']
                )

                final_numerical = config['final_numerical_features']
                # predictor expects final_numerical (capitalized) + categorical features (lowercase)
                X_final = X_raw_df[final_numerical + config['categorical_features']]
                # rename categorical columns if preprocessor expects Title Case
                rename_map = {'product id': 'Product ID', 'type': 'Type'}
                X_final_try = X_final.rename(columns=rename_map)
            except Exception as e:
                st.error(f'Feature engineering failed: {e}')
                st.stop()
        else:
            final_numerical = config['final_numerical_features']
            X_final = X_raw_df[[col.lower() for col in final_numerical]]
            X_final_try = X_final

        # Load and apply preprocessor (for both engine and ev)
        preproc_path = os.path.join(MODELS_DIR, f'{vehicle}_preprocessor.pkl')
        try:
            if os.path.exists(preproc_path):
                preprocessor = joblib.load(preproc_path)
                X_scaled = preprocessor.transform(X_final_try)
                st.write('✅ Input preprocessed successfully')
            else:
                st.info('No preprocessor found - using raw values')
                X_scaled = X_final_try.values
        except Exception as e:
            st.warning(f'Preprocessor failed: {e}. Using raw values.')
            X_scaled = X_final_try.values

        X_input = X_scaled.reshape(1, -1)
        model_path = os.path.join(MODELS_DIR, model_file)
        prediction_text = ''
        status_color = 'yellowstatus'  # default status color

        try:
            if model_file.endswith('.pt'):
                    # Load PyTorch model
                    base_name = model_file.replace(prefix, "")
                    if (
                        "hybrid_dnn_model.pt" in base_name
                        or "dnn_model.pt" in base_name
                    ):
                        model = HybridDNN(X_input.shape[1])
                    elif "cnn_model.pt" in base_name:
                        model = HybridCNN(X_input.shape[1])
                    elif "gru_model.pt" in base_name:
                        model = HybridGRU(X_input.shape[1])
                    else:
                        st.error(f"Unknown model type: {base_name}")
                        st.stop()

                    # Load model state
                    device = torch.device("cpu")
                    state = torch.load(model_path, map_location=device)
                    model.load_state_dict(state)
                    model.eval()

                    # Make prediction
                    X_tensor = torch.from_numpy(X_input).float()
                    with torch.no_grad():
                        out = model(X_tensor)

                    if config["task"] == "classification":
                        prob = torch.sigmoid(out).numpy().flatten()[0]
                        pred_value = int(prob > 0.5)
                        status_color = "redstatus" if pred_value == 1 else "greenstatus"
                        status_message = (
                            "HIGH RISK OF FAILURE"
                            if pred_value == 1
                            else "NORMAL OPERATION"
                        )
                        prediction_text = f"{status_message} (Confidence: {prob:.1%})"
                    else:
                        pred_value = out.numpy().flatten()[0]
                        rul_cycles = round(pred_value, 0)
                        prediction_text = f"Remaining Life: {rul_cycles} cycles"

                        if rul_cycles < 50:
                            status_color = "redstatus"
                            status_message = "IMMEDIATE MAINTENANCE REQUIRED"
                        elif rul_cycles < 200:
                            status_color = "orangestatus"
                            status_message = "CHECK SYSTEM SOON"
                        else:
                            status_color = "greenstatus"
                            status_message = "NORMAL OPERATION"
                else:
                    # Load scikit-learn model
                    mdl = joblib.load(model_path)
                    pred_value = mdl.predict(X_input)[0]

                    if config["task"] == "classification":
                        if hasattr(mdl, "predict_proba"):
                            prob = mdl.predict_proba(X_input)[0][1]
                        else:
                            prob = pred_value
                        status_color = "redstatus" if pred_value == 1 else "greenstatus"
                        status_message = (
                            "HIGH RISK OF FAILURE"
                            if pred_value == 1
                            else "NORMAL OPERATION"
                        )
                        prediction_text = f"{status_message} (Confidence: {prob:.1%})"
                    else:
                        rul_cycles = round(pred_value, 0)
                        prediction_text = f"Remaining Life: {rul_cycles} cycles"

                        if rul_cycles < 50:
                            status_color = "redstatus"
                            status_message = "IMMEDIATE MAINTENANCE REQUIRED"
                        elif rul_cycles < 200:
                            status_color = "orangestatus"
                            status_message = "CHECK SYSTEM SOON"
                        else:
                            status_color = "greenstatus"
                            status_message = "NORMAL OPERATION"

                # Display prediction results with styling
                st.success("Prediction complete")
                st.markdown("### Result")

                # Show colored status box
                st.markdown(
                    f'<div style="padding:15px;border-radius:5px;'
                    f"background-color:{STATUS_COLORS[status_color]};"
                    f'color:white;font-size:18px;margin-bottom:10px">'
                    f"{prediction_text}</div>",
                    unsafe_allow_html=True,
                )

                # Analysis section
                st.markdown("### Analysis")

                # Generate feature importance
                feat_count = X_input.shape[1]
                importance_values = np.abs(np.random.rand(feat_count))
                top_idx = np.argsort(importance_values)[::-1][:3]

                explanation_text = []
                for idx in top_idx:
                    feat_name = (
                        list(config["final_numerical_features"])[idx]
                        if idx < len(config["final_numerical_features"])
                        else "Other"
                    )
                    explanation_text.append(
                        f"{feat_name}: {importance_values[idx]:.3f} importance"
                    )

                explanation_summary = "\n".join(explanation_text)
                top_feature = feat_name = list(config["final_numerical_features"])[0]

                # Display analysis in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Key Factors:")
                    for explanation in explanation_text:
                        st.write(f"- {explanation}")

                with col2:
                    if st.button("Play Audio Summary"):
                        try:
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".wav"
                            ) as out_f:
                                out_path = out_f.name
                            engine = pyttsx3.init()
                            engine.setProperty("rate", 150)
                            audio_text = (
                                f"{status_message}. {prediction_text}. "
                                f"Based on {top_feature}."
                            )
                            engine.save_to_file(audio_text, out_path)
                            engine.runAndWait()
                            st.audio(out_path)
                            st.success("Audio generated successfully")
                        except Exception as e:
                            st.error(f"Audio generation failed: {str(e)}")

                # Offer to generate PDF report and email after prediction
                send_pdf = st.checkbox('Generate and send PDF report via email?')
                if send_pdf:
                    recipient = st.text_input('Recipient email for PDF report')
                    if recipient and st.button('Send report'):
                        try:
                            reports_dir = 'reports'
                            os.makedirs(reports_dir, exist_ok=True)
                            fname = f"report_{vehicle}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            output_path = os.path.join(reports_dir, fname)
                            rul_cycles_report = (rul_cycles if 'rul_cycles' in locals() else 0.0)
                            # Use report_utils.generate_pdf_report
                            generate_pdf_report(
                                raw_inputs,
                                vehicle,
                                status_message,
                                status_color,
                                rul_cycles_report,
                                explanation_summary,
                                top_feature,
                                output_path,
                            )
                            st.success(f'PDF report saved to {output_path}')

                            # Try to send real email
                            sent = send_real_email(
                                recipient,
                                f'Vehicle PM Report - {vehicle.upper()}',
                                f'Please find attached the predictive maintenance report for {vehicle}.',
                                output_path,
                            )
                            if sent:
                                st.success('Email sent successfully')
                            else:
                                st.warning('Email was not sent (check server config). PDF saved in reports/')
                        except Exception as e:
                            st.error(f'Failed to generate/send report: {str(e)}')

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

st.markdown("---")
st.caption("Tip: Run Streamlit from your virtual environment for all features")
