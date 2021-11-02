FROM python:3.8-slim

ENV PYTHONUNBUFFERED True

RUN python -m pip install --upgrade pip

COPY requirements.txt /
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000
ENV PORT 5000

WORKDIR /app

ENV API_KEY None

CMD exec gunicorn --bind :$PORT main:app --workers 1 --threads 1 --timeout 0
