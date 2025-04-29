import smtplib
from email.mime.text import MIMEText
import logging

import requests

def send_email(subject, body, to_email, smtp_server, smtp_port, username, password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = username
    msg['To'] = to_email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(username, [to_email], msg.as_string())
        return True
    except Exception as e:
        logging.error(f"Email alert failed: {e}")
        return False

def send_telegram(message, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    try:
        r = requests.post(url, data=data)
        return r.status_code == 200
    except Exception as e:
        logging.error(f"Telegram alert failed: {e}")
        return False
