#!/bin/bash

# Check and install dependencies
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "$1 not found. Installing..."
        sudo apt-get update
        sudo apt-get install -y "$2"
    fi
}

check_dependency screen screen
check_dependency redis-server redis-server
check_dependency python3 python3
check_dependency pip3 python3-pip

# Optional: Uncomment if dependencies aren't installed
# pip3 install celery redis

# Function to start a screen session
start_screen_session() {
    screen -dmS "$1" bash -c "$2"
    sleep 2  # Wait for session to start
}

# Start Redis first
start_screen_session redis "redis-server"
sleep 3  # Extra wait for Redis to initialize

# Start Celery worker
start_screen_session celery "celery -A monitor.tasks worker --loglevel=info"
sleep 2

# Start monitor
start_screen_session monitor "python3 run_monitor.py"
sleep 2

# Start listener
start_screen_session listener "python3 listener.py"

# List and verify screen sessions
echo "Screen sessions created:"
screen -ls
