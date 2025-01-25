import redis
import json
import os

def listen_violations():
    log_path = '/Users/vincent/Desktop/videoSummarization/logs/chat.log'
    
    client = redis.Redis(host='localhost', port=6379, db=0)
    pubsub = client.pubsub()
    pubsub.subscribe('violations')
    
    print("Listening for violations...")
    for message in pubsub.listen():
        if message['type'] == 'message':
            violation = json.loads(message['data'])
            print(f"\nViolation Detected!")
            print(f"Time: {violation['timestamp']}")
            print(f"Message: {violation['message']}")
            print(f"Violations: {', '.join(violation['violations'])}")

if __name__ == '__main__':
    listen_violations()