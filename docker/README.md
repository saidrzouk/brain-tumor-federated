# Docker setup

Runs the federated learning pipeline as containers: one Flower **server** and
three **clients** (one per hospital), plus an optional Gradio **app**.

| File | Purpose |
|------|---------|
| `Dockerfile.server` | Flower server (FedAvg). Light image, no PyTorch. |
| `Dockerfile.client` | Flower client. CPU PyTorch, VGG16 ImageNet weights baked in. |
| `Dockerfile.app` | Gradio app (expects `app/app.py` and trained weights). |
| `healthcheck.py` | Lets the clients wait until the server accepts connections. |
| `requirements-*.txt` | Minimal dependencies per image. |
| `../docker-compose.yml` | Starts server + 3 clients (and the app on request). |
| `../.dockerignore` | Keeps data, weights and results out of the images. |

## Prerequisites

1. Docker Desktop (or Docker Engine + the Compose plugin).
2. The dataset in `data/raw/` on your machine:

   ```bash
   python data/download_dataset.py
   ```

   The images are **mounted** into the client containers (read-only), not copied
   into the images.

## Run the federated training

From the project root:

```bash
docker compose up --build
```

The first build is slow (PyTorch + VGG16 weights, about 1.5 GB). Later builds
reuse the cache. The server waits for the 3 clients, then runs 10 FedAvg
rounds (see `federated/flower_config.py`). All containers stop when it ends.

Useful commands:

```bash
docker compose up --build -d      # run in the background
docker compose logs -f server     # follow the server (rounds, accuracy)
docker compose logs -f client1    # follow one client
docker compose ps                 # state of each container
docker compose down               # stop and remove the containers
```

## Run the app (optional)

```bash
docker compose --profile app up --build app
```

Then open <http://localhost:7860>. This needs the `app/` folder in the repo and
a trained model in `model/saved_models/` (mounted into the container).

## How it works

- Containers talk over the default Compose network. Clients reach the server at
  `server:8080` (the service name is the hostname).
- `SERVER_ADDRESS` is read by `federated/flower_config.py`. It is
  `0.0.0.0:8080` in the server container (listen on all interfaces) and
  `server:8080` in the client containers (where to connect). Without the
  variable it stays `127.0.0.1:8080` for local runs.
- `CLIENT_ID` (1 to 3) selects the client's data partition.
- Port 8080 is also published on your machine, so a client running outside
  Docker can connect to `localhost:8080`.

## Memory requirements (important)

Training VGG16 on CPU is memory hungry, and the 3 clients train **at the same
time**. Measured on a 16 GB Windows laptop with Docker Desktop (WSL2):

| Container | Memory seen |
|-----------|-------------|
| each client | about 2.3 GB at the start of training, **3.5 GB** when the kernel killed one |
| server | about 1.9 GB |

So the 4 containers need roughly **12.5 GB or more** for Docker alone. With
Docker limited to about 10 GB, one client was killed by the kernel
(`Out of memory: Killed process ... python` in `dmesg`) and restarted, and
the round could not complete with 3 clients.

Recommendations:

- Run the full 3-client training on a machine with **24 GB of RAM or more**
  (or give Docker at least 14 GB), or on Colab.
- On Windows (WSL2), the Docker memory limit is set in
  `C:\Users\<you>\.wslconfig`:

  ```ini
  [wsl2]
  memory=14GB
  swap=8GB
  ```

  then run `wsl --shutdown` and restart Docker Desktop. Check the limit with
  `docker run --rm alpine free -m`.
- On a smaller machine, use `python federated/run_simulation.py`, which trains
  the clients one after another instead of in parallel.

How to recognise the problem: a container shows `Exited (137)` or restarts in
`docker compose ps -a`, and the server log stops after `configure_fit`.
`docker inspect <container> --format "{{.State.OOMKilled}}"` may print
`false` even then, because the whole Docker VM ran out of memory. To confirm:

```bash
wsl -d docker-desktop dmesg | findstr /i "oom killed"
```

## Changing the number of clients

Add or remove `clientN` services in `docker-compose.yml`, and keep these in
sync: `NUM_CLIENTS` / `MIN_FIT_CLIENTS` in `federated/flower_config.py` and
`N_CLIENTS` in `data/split_data.py`. The server waits for `MIN_FIT_CLIENTS`
clients before starting.

## Troubleshooting

- **`FileNotFoundError ... data/raw/glioma`**: the dataset is missing. Run
  `python data/download_dataset.py` on the host and check that `data/raw/`
  contains the four class folders.
- **Containers killed, restarting or very slow**: see "Memory requirements"
  above. VGG16 on CPU is heavy and the 3 clients train in parallel.
- **Clients keep restarting**: look at `docker compose logs client1`. Clients
  retry up to 3 times if they fail.
- **GPU**: the images are CPU-only. A CUDA build needs a different
  `TORCH_INDEX_URL` build argument and the NVIDIA Container Toolkit (not tested here).