FROM --platform=$TARGETPLATFORM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install numpy blend_modes==2.2.0 opencv-python

COPY . .

RUN python -m pip install -e .

CMD ["python3", "-m", "unittest", "tests.test_blend_modes"]
