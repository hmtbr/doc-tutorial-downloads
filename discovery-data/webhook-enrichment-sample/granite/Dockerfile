FROM --platform=linux/amd64 python:3-alpine

WORKDIR /app

COPY requirements.txt /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm requirements.txt

COPY main.py /app

CMD ["python", "main.py"]
