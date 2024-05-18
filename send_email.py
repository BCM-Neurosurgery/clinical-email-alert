import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List
import os


class EmailSender:
    def __init__(self, smtp_server, smtp_port, smtp_user, smtp_password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.server = None

    def connect(self):
        self.server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        self.server.ehlo()
        self.server.starttls()
        self.server.login(self.smtp_user, self.smtp_password)

    def send_email(
        self,
        to_addrs: List[str],
        subject: str,
        body: str,
        attachments: List[str] = None,
    ):
        msg = MIMEMultipart()
        msg["From"] = self.smtp_user
        msg["To"] = ", ".join(to_addrs)
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        if attachments:
            for file in attachments:
                attachment = open(file, "rb")
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {os.path.basename(file)}",
                )
                msg.attach(part)

        try:
            self.server.sendmail(self.smtp_user, to_addrs, msg.as_string())
            return "Email sent successfully with attachments"
        except Exception as e:
            return f"Failed to send email: {str(e)}"

    def disconnect(self):
        if self.server:
            self.server.quit()
