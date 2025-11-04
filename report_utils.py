import os
import smtplib
from email.message import EmailMessage

try:
    from fpdf import FPDF, XPos, YPos

    FPDF_AVAILABLE = True
except Exception:
    FPDF = None
    XPos = None
    YPos = None
    FPDF_AVAILABLE = False
from datetime import datetime
from typing import Dict, Union
from dotenv import load_dotenv

load_dotenv()  # this loads the variables from your .env into os.environ


# --- CONFIGURATION (UPDATE THESE FOR REAL EMAIL) ---
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "your_sender_email@gmail.com")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "your_app_password_here")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
# ----------------------------------------------------

# PDF Report Constants
COLOR_BLUE = (30, 144, 255)
COLOR_GRAY = (240, 240, 240)
COLOR_GREEN = (34, 139, 34)
COLOR_ORANGE = (255, 165, 0)
COLOR_RED = (220, 20, 60)


def speak(text):
    print(f"🗣️ {text}")


def generate_suggestions(status: str, top_feature: str) -> list[str]:
    base_suggestions = [
        f'Inspect sensor readings for {top_feature.replace("_", " ")} for calibration drift or anomalies.',
        "Perform a complete systems audit (fluids, filters, connections) to rule out minor issues.",
        f'Verify the integrity and calibration of all components related to the {top_feature.replace("_", " ")} measurement.',
    ]

    if "HIGH RISK" in status or "Immediate maintenance" in status:
        critical_suggestions = [
            "URGENT: Isolate the unit and proceed with component replacement or major repair planning.",
            "Schedule immediate diagnostic testing focused on the system that generated high readings.",
        ]
        return critical_suggestions + base_suggestions

    if "Check system soon" in status:
        caution_suggestions = [
            "Schedule a preventive maintenance window for early intervention.",
        ]
        return caution_suggestions + base_suggestions

    return [
        "Review historical data trends to detect subtle degradation."
    ] + base_suggestions


def handle_report_and_email(
    raw_input_dict: Dict[str, Union[str, float]],
    vehicle_choice_key: str,
    status_message: str,
    status_color: str,
    rul_cycles: float,
    explanation_summary: str,
    top_feature: str,
) -> None:
    """Generate a PDF report and optionally email it to the user.

    Args:
        raw_input_dict: Dictionary of input feature values
        vehicle_choice_key: Type of vehicle ('engine' or 'ev')
        status_message: Predicted status/health message
        status_color: Color code for status visualization
        rul_cycles: Remaining useful life in cycles (may be 0.0 for classification)
        explanation_summary: SHAP/LIME explanation text
        top_feature: Most influential feature name
    """
    speak(
        "Starting professional predictive maintenance report generation (PDF format)."
    )

    if not FPDF_AVAILABLE:
        speak(
            "PDF report generation requires the 'fpdf2' package. "
            "Install it with: pip install fpdf2"
        )
        return

    REPORTS_DIR = "reports"
    os.makedirs(REPORTS_DIR, exist_ok=True)

    output_filename_base = (
        f"report_{vehicle_choice_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    output_path = os.path.join(REPORTS_DIR, output_filename_base)

    try:
        pdf_path = generate_pdf_report(
            raw_input_dict,
            vehicle_choice_key,
            status_message,
            status_color,
            rul_cycles,
            explanation_summary,
            top_feature,
            output_path,
        )
        speak(f"✅ PDF Report successfully created and saved to: '{pdf_path}'.")
    except Exception as e:
        speak(f"❌ ERROR: PDF report generation failed: {e}")
        speak(
            "Ensure 'fpdf2' is installed (pip install fpdf2) and all parameters are correct."
        )
        return

    recipient_email = input(
        "Please manually enter the recipient email for the report: "
    )
    if not ("@" in recipient_email and "." in recipient_email.split("@")[1]):
        recipient_email = "default_recipient@company.com"
        speak(f"Invalid email provided. Using default: {recipient_email}")

    speak(f"Recipient: {recipient_email}. Attempting real email dispatch.")

    subject = f"Vehicle PM Report - {status_message.split('(')[0].strip()} ({vehicle_choice_key.upper()})"
    email_body = (
        f"Dear Team,\n\nA Predictive Maintenance Report has been generated "
        f"for a {vehicle_choice_key.upper()} unit.\n\n"
        f"STATUS: {status_message.replace('**', '')}\n"
        f"Influential Factor: {top_feature.replace('_', ' ')}\n\n"
        f"Please find the full diagnostic report attached as a PDF file."
    )

    send_real_email(recipient_email, subject, email_body, pdf_path)
    speak(
        f"✅ Process complete. Check your reports folder for '{output_filename_base}'."
    )


def send_real_email(
    recipient: str, subject: str, body: str, attachment_path: str
) -> bool:
    """Send an email with an optional PDF attachment. Returns True on success.

    The function uses environment variables SENDER_EMAIL and SENDER_PASSWORD. If
    these are left as the default placeholders the function will warn and return False.
    """
    if (
        SENDER_EMAIL == "your_sender_email@gmail.com"
        or SENDER_PASSWORD == "your_app_password_here"
    ):
        speak(
            "❌ WARNING: SENDER_EMAIL or SENDER_PASSWORD not configured. Skipping real email send."
        )
        speak(
            "Please set SENDER_EMAIL and SENDER_PASSWORD environment variables (use an App Password for Gmail)."
        )
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient
    msg.set_content(body)

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_path)
        msg.add_attachment(
            file_data, maintype="application", subtype="pdf", filename=file_name
        )

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        speak(f"✅ Real email successfully sent to {recipient} with PDF attachment!")
        return True
    except smtplib.SMTPAuthenticationError:
        speak(
            "❌ ERROR: Email failed. Check your SENDER_EMAIL/SENDER_PASSWORD (use App Password for Gmail)."
        )
    except Exception as e:
        speak(f"❌ ERROR: Failed to send email: {e}")

    return False


if FPDF_AVAILABLE:

    class MaintenancePDF(FPDF):
        """Custom FPDF class for the professional report structure."""

        def header(self):
            self.set_fill_color(*COLOR_BLUE)
            self.rect(0, 0, 210, 20, "F")

            self.set_font("Arial", "B", 16)
            self.set_text_color(255, 255, 255)
            self.cell(
                0,
                10,
                "Predictive Maintenance Diagnostic Report",
                new_y=YPos.NEXT,
                align="C",
                fill=False,
            )
            self.set_font("Arial", "", 10)
            self.cell(
                0,
                5,
                datetime.now().strftime("Report Generated: %Y-%m-%d %H:%M:%S"),
                new_y=YPos.NEXT,
                align="C",
            )
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

        def chapter_title(self, title):
            self.set_font("Arial", "B", 14)
            self.set_text_color(*COLOR_BLUE)
            self.cell(0, 10, title, new_y=YPos.NEXT, align="L")
            self.line(
                self.get_x(), self.get_y() - 1, self.w - self.r_margin, self.get_y() - 1
            )
            self.ln(5)
            self.set_text_color(0, 0, 0)

        def status_box(
            self, status_message, status_color_name, rul_cycles, vehicle_type
        ):
            if "HIGH RISK" in status_message or "Immediate" in status_message:
                color = COLOR_RED
            elif "Check system soon" in status_message:
                color = COLOR_ORANGE
            else:
                color = COLOR_GREEN

            self.set_fill_color(*color)
            self.set_text_color(255, 255, 255)
            self.set_font("Arial", "B", 18)

            self.set_line_width(0)
            self.rect(
                10, self.get_y(), 190, 20, "F"
            )  # removed unsupported round_corners
            self.set_xy(10, self.get_y() + 5)

            status_text = status_message.replace("**", "").replace("\n", " ")
            # Wrap long status if needed
            max_width = 95
            if self.get_string_width(status_text) > max_width:
                self.set_font("Arial", "B", 14)
            self.cell(100, 10, f"CURRENT STATUS: {status_text}", align="L", fill=False)

            if vehicle_type == "ev":
                self.set_xy(110, self.get_y())
                self.set_font("Arial", "B", 14)
                self.cell(
                    90, 10, f"RUL: {int(rul_cycles):,} cycles", align="R", fill=False
                )

            self.ln(20)
            self.set_text_color(0, 0, 0)
            self.set_font("Arial", "", 10)
            self.ln(5)

        def data_table(self, raw_input: Dict[str, Union[str, float]]):
            self.chapter_title("1. Input Data Snapshot")
            col_widths = [90, 100]
            self.set_fill_color(*COLOR_GRAY)
            self.set_font("Arial", "B", 10)
            self.cell(
                col_widths[0],
                7,
                "Parameter",
                1,
                new_x=XPos.RIGHT,
                new_y=YPos.TOP,
                align="L",
                fill=True,
            )
            self.cell(
                col_widths[1],
                7,
                "Value",
                1,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
                align="L",
                fill=True,
            )
            self.set_font("Arial", "", 10)

            for idx, (name, value) in enumerate(raw_input.items()):
                safe_name = name.replace("_", " ")
                safe_value = str(value)
                self.set_fill_color(255, 255, 255)
                if idx % 2 == 1:
                    self.set_fill_color(*COLOR_GRAY)
                self.cell(
                    col_widths[0],
                    6,
                    safe_name,
                    "LR",
                    new_x=XPos.RIGHT,
                    new_y=YPos.TOP,
                    align="L",
                    fill=True,
                )
                self.cell(
                    col_widths[1],
                    6,
                    safe_value,
                    "LR",
                    new_x=XPos.LMARGIN,
                    new_y=YPos.NEXT,
                    align="R",
                    fill=True,
                )

            self.cell(
                sum(col_widths),
                0,
                "",
                "T",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
                align="C",
            )
            self.ln(5)

        def suggestions_list(self, suggestions: list[str]):
            self.chapter_title("3. Actionable Maintenance Suggestions")
            self.set_font("Arial", "", 10)
            for item in suggestions:
                self.set_text_color(*COLOR_BLUE)
                self.cell(5, 7, "•", 0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="L")
                self.set_text_color(0, 0, 0)
                if "URGENT" in item or "immediate diagnostic" in item:
                    self.set_font("Arial", "B", 10)
                    self.set_text_color(*COLOR_RED)
                self.multi_cell(
                    self.w - 2 * self.l_margin, 7, item, 0, "L"
                )  # fixed width
                self.set_font("Arial", "", 10)

    def generate_pdf_report(
        raw_input: Dict[str, Union[str, float]],
        vehicle_type: str,
        status_message: str,
        status_color: str,
        rul_cycles: float,
        explanation_summary: str,
        top_feature: str,
        output_path: str,
    ) -> str:

        pdf = MaintenancePDF("P", "mm", "A4")

        try:
            pdf.add_font("Arial", "", os.path.join("Font", "arial.ttf"), uni=True)
            pdf.add_font("Arial", "B", os.path.join("Font", "arialbd.ttf"), uni=True)
            pdf.add_font("Arial", "I", os.path.join("Font", "ariali.ttf"), uni=True)
        except Exception:
            pass

        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.alias_nb_pages()
        pdf.add_page()

        pdf.status_box(status_message, status_color, rul_cycles, vehicle_type)
        pdf.data_table(raw_input)

        pdf.chapter_title("2. Model Explanation and Key Factor")
        pdf.set_font("Arial", "", 10)

        explanation_summary = explanation_summary.replace(
            top_feature, f'***{top_feature.replace("_", " ")}***'
        )
        explanation_summary = explanation_summary.replace("**", "")
        pdf.set_x(pdf.l_margin)  # start at left margin
        pdf.multi_cell(
            pdf.w - 2 * pdf.l_margin, 5, explanation_summary
        )  # fixed width to avoid error

        pdf.set_font("Arial", "B", 10)
        pdf.ln(3)
        pdf.cell(
            0,
            5,
            f'Most Influential Feature: {top_feature.replace("_", " ")}',
            new_y=YPos.NEXT,
            align="L",
        )
        pdf.ln(5)

        suggestions_list_data = generate_suggestions(status_message, top_feature)
        pdf.suggestions_list(suggestions_list_data)

        pdf.output(output_path)
        return output_path

else:
    # fpdf2 not available — provide fallbacks that inform the caller
    def generate_pdf_report(*args, **kwargs):
        raise RuntimeError(
            "PDF generation requires the 'fpdf2' package. Install it with 'pip install fpdf2'."
        )
