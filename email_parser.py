import email
import extract_msg
from email import policy

def parse_eml(file_content):
    
    try:
        msg = email.message_from_bytes(file_content, policy=policy.default)
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode("utf-8", errors="ignore")
        else:
            return msg.get_payload(decode=True).decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error parsing .eml file: {str(e)}"
    return ""

def parse_msg(file):
   
    try:
        msg = extract_msg.Message(file)
        body = msg.body
        msg.close()
        return body
    except Exception as e:
        return f"Error parsing .msg file: {str(e)}"
    return ""