FROM python:3.9-slim-buster

WORKDIR /app

COPY app.py .

RUN pip install flask pillow requests transformers torch torchvision torchaudio

EXPOSE 5000

CMD ["python", "app.py"]