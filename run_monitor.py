# monitor.py
from celery import Celery
import time
from monitor.tasks import analyze_message

def monitor_chat_log():
    log_path = '/Users/vincent/Desktop/videoSummarization/logs/chat.log'
    with open(log_path, 'r') as f:
        f.seek(0, 2)  # Move to end of file
        while True:
            line = f.readline()
            if line:
                analyze_message.delay(line)
            time.sleep(1)

if __name__ == '__main__':
    monitor_chat_log()