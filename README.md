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

## Docker

The federated pipeline runs as containers on the shared network `fl_network`:
`server` (port 8080), `hospital_1`, `hospital_2`, `hospital_3` and `webapp`
(port 7860). Requirements: Docker Desktop (or Docker Engine + Compose) and the
dataset in `data/raw/`.

```bash
python data/download_dataset.py   # once, on your computer
docker compose up --build         # server + 3 hospitals + web app
```

The MRI images are mounted (read-only) into the hospital containers, not copied
into the images. Hospitals wait for the server to be healthy. Clients reach it
at `server:8080`; `SERVER_ADDRESS` is read by `federated/flower_config.py`
(default `127.0.0.1:8080` for local runs) and `CLIENT_ID` (1 to 3) selects the
hospital's data partition.

Useful commands:

```bash
docker compose logs -f server      # follow the federated rounds
docker compose logs -f hospital_1  # follow one hospital
docker compose ps                  # state of each container
docker compose down                # stop and remove the containers
```

The web app needs the `app/` folder (`app/app.py`) and a trained model in
`model/saved_models/`. Until `app/` exists, its image cannot be built, so start
only the federated part:

```bash
docker compose up --build server hospital_1 hospital_2 hospital_3
```

### Memory requirements

Training VGG16 on CPU is memory hungry, and the 3 hospitals train at the same
time. Measured on a 16 GB Windows laptop with Docker Desktop (WSL2): each client
used about 2.3 GB at the start of training and 3.5 GB when the kernel killed
one; the server used about 1.9 GB. The 4 containers need roughly **12.5 GB or
more** for Docker alone. With Docker limited to about 10 GB, one client was
killed (`Out of memory: Killed process ... python`) and the round could not
complete with 3 clients.

- Run the full training on a machine with 24 GB of RAM or more (or give Docker
  at least 14 GB), or on Colab.
- On Windows (WSL2) set the limit in `C:\Users\<you>\.wslconfig`
  (`[wsl2]`, `memory=14GB`, `swap=8GB`), then run `wsl --shutdown` and restart
  Docker Desktop. Check it with `docker run --rm alpine free -m`.
- On a smaller machine use `python federated/run_simulation.py`, which trains
  the clients one after another.

Signs of the problem: a container shows `Exited (137)` or restarts in
`docker compose ps -a`, and the server log stops after `configure_fit`.
`docker inspect <container> --format "{{.State.OOMKilled}}"` may print `false`
even then, because the whole Docker VM ran out of memory. To confirm:
`wsl -d docker-desktop dmesg | findstr /i "oom killed"`.

### Changing the number of hospitals

Add or remove `hospital_N` services in `docker-compose.yml`, and keep these in
sync: `NUM_CLIENTS` in `federated/flower_config.py` and `N_CLIENTS` in
`data/split_data.py`. `MIN_FIT_CLIENTS`, `MIN_EVALUATE_CLIENTS` and
`MIN_AVAILABLE_CLIENTS` (2 by default) are the minimum number of hospitals the
server needs: with 2 out of 3, training continues when one hospital stops
(`docker compose stop hospital_2`), and the hospital can rejoin later.

### Troubleshooting

- `FileNotFoundError ... data/raw/glioma`: the dataset is missing. Run
  `python data/download_dataset.py` on the host.
- Containers killed, restarting or very slow: see "Memory requirements".
- Hospitals keep restarting: `docker compose logs hospital_1` (they retry up to
  3 times).
- GPU: the images are CPU-only. A CUDA build needs another `TORCH_INDEX_URL`
  build argument and the NVIDIA Container Toolkit (not tested here).

## Testing

Run the test suite with:

```bash
pytest
```

