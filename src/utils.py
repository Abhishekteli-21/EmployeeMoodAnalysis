import hashlib
import yaml
import numpy as np
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def anonymize_employee_id(employee_id, key, use_timestamp=True):
    salt = datetime.now().isoformat() if use_timestamp else ""
    return hashlib.sha256(f"{employee_id}{key}{salt}".encode()).hexdigest()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def send_email(to_email, subject, body, config):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = config["notifications"]["sender_email"]
    msg["To"] = to_email

    try:
        with smtplib.SMTP(config["notifications"]["smtp_server"], config["notifications"]["smtp_port"]) as server:
            server.starttls()
            server.login(config["notifications"]["sender_email"], config["notifications"]["sender_password"])
            server.send_message(msg)
            print(f"Email sent to {to_email}: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")