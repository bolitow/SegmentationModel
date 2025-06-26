#!/bin/bash

# Start FastAPI backend on $PORT
cd /app/api && uvicorn main:app --host 0.0.0.0 --port $PORT

# Exit with status of process
exit $?