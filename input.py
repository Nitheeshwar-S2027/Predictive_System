import os
import re
import string
import tempfile
from typing import Union, List, Dict

# Optional TTS (pyttsx3) - may not be installed in all environments
try:
    import pyttsx3

    TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None
    TTS_AVAILABLE = False

# Optional STT (faster-whisper, sounddevice, soundfile, word2number)
try:
    from faster_whisper import WhisperModel
    import sounddevice as sd
    import soundfile as sf
    from word2number import w2n

    STT_AVAILABLE = True
except Exception:
    # Missing optional STT deps: app will still work in text mode
    WhisperModel = None
    sd = None
    sf = None
    w2n = None
    STT_AVAILABLE = False


# ------------------ SPEAK FUNCTION ------------------
def speak_local(text: str):
    """Speak the given text aloud and print it."""
    print(f"🗣️ {text}")
    try:
        if TTS_AVAILABLE and pyttsx3 is not None:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
    except Exception:
        pass


# ------------------ WHISPER MODEL INITIALIZATION ------------------
WHISPER_MODEL_SIZE = "small.en"
stt_model = None


def initialize_voice_model():
    """Initializes Whisper model when using voice mode."""
    global stt_model
    if not STT_AVAILABLE:
        print("❌ Voice input not available. Install dependencies first.")
        return False
    try:
        speak_local("🎧 Loading Faster-Whisper voice model...")
        stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu")
        speak_local(f"✅ Loaded Faster-Whisper ({WHISPER_MODEL_SIZE}) successfully.")
        return True
    except Exception as e:
        print(f"❌ Could not load Whisper model: {e}")
        return False


# ------------------ LISTEN FUNCTION ------------------
def listen(duration: int = 5) -> str:
    """Captures voice input and returns transcribed text."""
    try:
        fs = 16000
        print("🎤 Listening, please speak now...")
        recording = sd.rec(
            int(duration * fs), samplerate=fs, channels=1, dtype="float32"
        )
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, recording, fs)
        segments, _ = stt_model.transcribe(f.name)
        os.remove(f.name)
        text = " ".join([seg.text for seg in segments]).lower().strip()
        print(f"Transcript: '{text}'")  # ADD THIS DEBUG LINE!
        # --- CRITICAL FIX HERE ---
        # Define punctuation to remove: all standard punctuation EXCEPT the period '.'
        punc_to_remove = string.punctuation.replace(".", "")

        # Remove only specified punctuation
        # This keeps the decimal point, but removes trailing commas, question marks, etc.
        return text.translate(str.maketrans("", "", punc_to_remove))
    except Exception as e:
        print(f"❌ Error during voice input: {e}")
        return ""


# ------------------ NUMBER PARSING ------------------
def parse_number(text: str) -> Union[float, None]:
    text = text.lower().replace("-", " ").replace(",", " ").strip()
    text = re.sub(r"\b(point|dot)\b", ".", text)

    tokens = []
    for t in text.split():
        t = t.strip(string.punctuation)

        try:
            if not re.search(r"\d|\.", t):
                tokens.append(str(w2n.word_to_num(t)))
                continue
        except Exception:
            pass

        if re.fullmatch(r"\d*\.\d+|\d+|\.", t):
            tokens.append(t)
        elif t in ("oh", "o", "zero"):
            tokens.append("0")

    clean_text = "".join(tokens)

    if clean_text.count(".") > 1:
        parts = clean_text.split(".", 1)
        clean_text = parts[0] + "." + parts[1].replace(".", "")

    clean_text = re.sub(r"[^\d.]", "", clean_text)

    if re.fullmatch(r"^\d*\.?\d+$", clean_text):
        try:
            return float(clean_text)
        except ValueError:
            return None

    return None


# ------------------ MAIN INPUT DATA COLLECTION ------------------
def get_input_data(
    speak_func,
    listen_func,
    parse_func,
    input_features: List[str],
    vehicle_choice_key: str = "",
    input_mode: str = "text",
) -> Dict[str, Union[str, float]]:
    """
    Collects input data based on selected mode ('voice' or 'text').

    Parameters:
    - speak_func: function to speak text
    - listen_func: function to capture voice input
    - parse_func: function to convert text to numbers
    - input_features: list of feature names
    - vehicle_choice_key: 'engine' or 'ev' (used for product ID)
    - input_mode: 'voice' or 'text'
    """
    raw_input_dict = {}

    # ---------------- TEXT MODE ----------------
    if input_mode == "text":
        print("\n🧠 TEXT MODE selected.")
        # Product ID input (if applicable)
        if "Product ID" in input_features and vehicle_choice_key == "engine":
            val = ""
            while val not in ["L", "M", "H"]:
                val = input("Enter product quality variant (L, M, H): ").strip().upper()
            raw_input_dict["Product ID"] = val

        # Numeric input all at once
        numeric_features = [f for f in input_features if f != "Product ID"]
        if numeric_features:
            while True:
                print(
                    f"\nEnter values for these features (comma-separated):\n{numeric_features}"
                )
                inp = input("➡ Values: ").strip()
                values = [v.strip() for v in inp.split(",")]
                if len(values) != len(numeric_features):
                    print(
                        f"❌ Expected {len(numeric_features)} values, got {len(values)}. Try again."
                    )
                    continue
                try:
                    numeric_values = [float(v) for v in values]
                    break
                except ValueError:
                    print("❌ Invalid numeric values. Please enter valid numbers.")
            raw_input_dict.update(dict(zip(numeric_features, numeric_values)))

        print(f"\n✅ Final input list: {list(raw_input_dict.values())}")
        return raw_input_dict

    # ---------------- VOICE MODE ----------------
    elif input_mode == "voice":
        # if not initialize_voice_model():
        #     print("⚠️ Falling back to text mode.")
        #     return get_input_data(speak_func, listen_func, parse_func, input_features, vehicle_choice_key, "text")

        speak_func(f"Starting voice input for {len(input_features)} features.")
        for feat in input_features:
            while True:
                # Product ID handling
                if feat == "Product ID" and vehicle_choice_key == "engine":
                    speak_func(
                        "Please say the product quality variant: Low, Medium, or High."
                    )
                    val_text = listen_func().upper()
                    if val_text.startswith("L"):
                        raw_input_dict[feat] = "L"
                        speak_func("You said Low. Confirmed.")
                        break
                    elif val_text.startswith("M"):
                        raw_input_dict[feat] = "M"
                        speak_func("You said Medium. Confirmed.")
                        break
                    elif val_text.startswith("H"):
                        raw_input_dict[feat] = "H"
                        speak_func("You said High. Confirmed.")
                        break
                    else:
                        speak_func("Invalid input. Please repeat Low, Medium, or High.")
                else:
                    unit_match = re.search(r"\[(.*?)\]", feat)
                    unit = unit_match.group(1) if unit_match else ""
                    speak_func(f"Please say the value for {feat} ({unit}).")
                    val_text = listen_func()
                    val = parse_func(val_text)
                    if val is not None:
                        speak_func(
                            f"I recorded {val} {unit}. Say 'yes' to confirm or 'no' to repeat."
                        )
                        confirm = listen_func()
                        if "yes" in confirm or "yeah" in confirm:
                            raw_input_dict[feat] = val
                            break
                        else:
                            speak_func(f"Let's repeat {feat}.")
                    else:
                        speak_func(
                            f"Could not recognize a valid number. Let's repeat {feat}."
                        )
        print(f"\n✅ Final input list: {list(raw_input_dict.values())}")
        return raw_input_dict

    else:
        print("❌ Invalid input mode. Use 'voice' or 'text'.")
        return {}


# ------------------ EXPORTS ------------------
__all__ = ["get_input_data", "listen", "parse_number", "speak_local"]
