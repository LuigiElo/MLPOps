FROM python:3.11-slim
EXPOSE $PORT
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt \
    fastapi \
    pydantic \
    uvicorn

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libtiff-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1 \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#RUN pip install --no-cache-dir --upgrade -r requirements.txt
#RUN pip install fastapi
#RUN pip install pydantic
#RUN pip install uvicorn

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

COPY / src/

# COPY path_to_saved_model.pth /app/
#COPY models/model.pth /app/
COPY mlsopsbasic/predict_model.py predict_model.py

#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
CMD exec uvicorn predict_model:app --port $PORT --host 0.0.0.0 --workers 1
