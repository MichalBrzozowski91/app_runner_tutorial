FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r app_requirements.txt

CMD ["python", "app.py"]