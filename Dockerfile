FROM tiangolo/uvicorn-gunicorn:python3.8-slim

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r ./requirements.txt
COPY . ./


# RUN chmod +x entrypoint.sh
# CMD ["./entrypoint.sh"]

# CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5001", "app.main:app", "--reload", "--workers","4"]