FROM python:3.12-slim
EXPOSE $PORT

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_backend.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlsopsbasic/ mlsopsbasic/
COPY models/ models/

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /
RUN pip install --prefer-binary -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

EXPOSE $PORT
#CMD exec uvicorn my_application:app --port $PORT --workers 1 main:app
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
CMD exec uvicorn mlsopsbasic.predict_model:app --port $PORT --host 0.0.0.0 --workers 1
