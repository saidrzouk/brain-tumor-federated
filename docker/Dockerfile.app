# Gradio web app for predictions.
# NOTE: assumes the entry point is app/app.py. Adjust CMD if your teammate
# names it differently. The trained weights are mounted at run time from
# model/saved_models (see docker-compose.yml).
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

RUN pip install "torch>=2.2,<3" "torchvision>=0.17,<1" --index-url ${TORCH_INDEX_URL}
COPY docker/requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY data/ ./data/
COPY model/ ./model/
COPY app/ ./app/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 7860

CMD ["python", "app/app.py"]
