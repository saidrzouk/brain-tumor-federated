# Gradio web app for predictions.
# NOTE: assumes the entry point is app/app.py. Adjust CMD if your teammate
# names it differently. The trained weights are mounted at run time from
# model/saved_models (see docker-compose.yml).
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

RUN pip install "torch>=2.2,<3" "torchvision>=0.17,<1" --index-url ${TORCH_INDEX_URL}
# scikit-learn: data/split_data.py imports it at module level
RUN pip install "gradio>=4.0,<7" "numpy>=1.26,<3" "Pillow>=10,<12" "scikit-learn>=1.4,<2"

COPY data/ ./data/
COPY model/ ./model/
COPY app/ ./app/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 7860

CMD ["python", "app/app.py"]