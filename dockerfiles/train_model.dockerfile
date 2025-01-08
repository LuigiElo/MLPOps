# Base image
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlsopsbasic/ mlsopsbasic/
COPY data/ data/

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN --mount=type=cache,target=~/pip/.cache pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mlsopsbasic/train_model.py","train"]
