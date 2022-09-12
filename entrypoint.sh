#!/bin/bash
RUN_PORT=${PORT:-8080}
/usr/local/bin/gunicorn --worker-tmp-dir /dev/shm --config gunicorn.config.py app.main:app --bind "0.0.0.0:${RUN_PORT}"