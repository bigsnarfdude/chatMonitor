from celery import Celery
from datetime import datetime
import redis
import json

app = Celery('monitor', broker='redis://localhost:6379/0')
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.task(name='monitor.analyze_message')
def analyze_message(message):
    offensive_words = ['faggot', 'slur', 'ass']
    violations = [word for word in offensive_words if word in message.lower()]

    if violations:
        violation_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'violations': violations
        }
        redis_client.publish('violations', json.dumps(violation_data))
        return True
    return False
