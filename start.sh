#!/bin/bash
echo "----------------------------------------------------------------"
echo "Create start.sh script"
echo "Container started at $(date)"
echo "User: $(whoami)"
echo "PWD: $(pwd)"
echo "LS of /app:"
ls -la /app
echo "----------------------------------------------------------------"

echo "Checking python version..."
/usr/bin/python3.11 --version || echo "PYTHON 3.11 NOT FOUND"

echo "Checking environment variables..."
env

echo "----------------------------------------------------------------"
echo "Starting handler..."
/usr/bin/python3.11 -u handler.py
EXIT_CODE=$?
echo "Handler exited with code $EXIT_CODE"
exit $EXIT_CODE
