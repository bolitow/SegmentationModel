#!/bin/bash

# Start FastAPI backend on $PORT
cd /app/api && uvicorn main:app --host 0.0.0.0 --port $PORT &

# Start Streamlit frontend on port 8501
cd /app/app && streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?