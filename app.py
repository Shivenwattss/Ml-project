import streamlit as st
import csv
import os
from datetime import datetime

from kmeans import run_kmeans
from DBSCANS import run_dbscan
from recommender import run_recommender

st.set_page_config(page_title="Clustering Lab", layout="wide")
st.title("Clustering Lab")

algo = st.sidebar.selectbox(
    "Choose Page",
    ["K-Means", "DBSCAN", "Algorithm Recommender"]
)

if algo == "K-Means":
    run_kmeans()
elif algo == "DBSCAN":
    run_dbscan()
else:
    run_recommender()

# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.markdown("---")
st.markdown("### 💬 Feedback")
st.caption(f"How helpful was the **{algo}** " + ("page?" if algo == "Algorithm Recommender" else "visualizer?"))

# 5-star rating
rating = st.feedback(
    "stars",
    key=f"rating_{algo}"
)

# Email
email = st.text_input(
    "📧 Your Email ID",
    placeholder="example@gmail.com",
    key=f"email_{algo}"
)

# Written feedback
comment = st.text_area(
    "📝 Your Feedback",
    placeholder="Tell us what you liked or what could be improved...",
    key=f"comment_{algo}"
)

if st.button("📩 Submit Feedback", type="primary"):

    # Validation
    if rating is None:
        st.warning("⭐ Please select a star rating.")

    elif not email:
        st.warning("📧 Please enter your email ID.")

    elif not comment.strip():
        st.warning("📝 Please write some feedback.")

    else:

        stars = rating + 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 1. SAVE TO CSV ──────────────────────────────────────────────────
        # NOTE: on Streamlit Community Cloud this file is wiped whenever the app
        # restarts/redeploys, so the email below is your permanent record.
        saved = False
        try:
            feedback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.csv")
            file_exists = os.path.isfile(feedback_path)
            with open(feedback_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "algorithm", "stars", "email", "comment"])
                writer.writerow([timestamp, algo, stars, email, comment])
            saved = True
        except Exception:
            pass

        # ── 2. SEND EMAIL ───────────────────────────────────────────────────
        try:
            sender_email = st.secrets["EMAIL_ADDRESS"]
            sender_password = st.secrets["EMAIL_PASSWORD"]
            receiver_email = st.secrets["RECEIVER_EMAIL"]

            subject = f"New Feedback - {algo} ⭐ {stars}/5"
            body = f"""
You received new feedback for your Clustering Lab.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Page Used:
{algo}

Rating:
{"⭐" * stars} ({stars}/5)

User Email:
{email}

Feedback:
{comment}

Timestamp:
{timestamp}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)

            st.success("✅ Thank you! Your feedback has been submitted.")

        except Exception as e:
            if saved:
                st.error("❌ Feedback was saved, but the email could not be sent.")
            else:
                st.error("❌ Feedback could not be saved or emailed.")
            st.error(f"Error: {e}")
