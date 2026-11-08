# Brain Tumor Federated Learning

Federated learning project for classifying brain MRI images into four classes:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The project is designed to compare centralized training with federated
learning under IID and non-IID data distributions.

## Project structure

```text
data/          Dataset download and IID/non-IID splitting
model/         VGG16 model, centralized training, and evaluation
federated/     Flower client, server, and configuration
app/           Gradio web app and CLI prediction
experiments/   IID, non-IID, centralized runs, and comparisons
tests/         Unit and integration tests
docker/        Dockerfiles for the server, clients, and app
report/        Final report and presentation
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the project requirements:

```bash
pip install -r requirements.txt
```

## Dataset

Raw MRI images should be placed in:

```text
data/raw/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/
```

Raw data, trained model weights, experiment outputs, and report PDFs are
excluded from Git using `.gitignore`.

## Experiments

The planned experiment entry points are:

```bash
python experiments/run_centralized.py
python experiments/run_iid.py
python experiments/run_noniid.py
python experiments/compare_results.py
```

## Testing

Run the test suite with:

```bash
pytest
```

