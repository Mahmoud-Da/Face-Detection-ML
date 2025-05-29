# MTCNN Face Detector Initialization

This Python script provides a utility function to initialize a pre-trained MTCNN (Multi-task Cascaded Convolutional Networks) face detector using the `facenet-pytorch` library. The detector's parameters are configurable via an external `config.py` file.

## Features

- Initializes an MTCNN model for face detection.
- Uses the robust `facenet-pytorch` implementation.
- All MTCNN parameters are configurable through a `config.py` file.
- Supports running on either CPU or GPU (CUDA) by specifying the device in `config.py`.
- Includes a simple example usage in the `if __name__ == '__main__':` block to verify model loading.

# Running Your PyTorch Project with Docker

This document outlines the steps to build and run this PyTorch application using Docker and Docker Compose. This ensures a consistent and reproducible environment for development and deployment.

## Prerequisites

1.  **Docker**: Ensure Docker Desktop (for Mac/Windows) or Docker Engine (for Linux) is installed and running. You can download it from [docker.com](https://www.docker.com/products/docker-desktop/).
2.  **Docker Compose**: Docker Compose V2 is typically included with Docker Desktop. For Linux, you might need to install it separately.
3.  **(Optional) NVIDIA GPU Support**:
    - If you intend to use NVIDIA GPUs, ensure you have the latest NVIDIA drivers installed on your host machine.
    - Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your host machine. This allows Docker containers to access NVIDIA GPUs.
4.  **Project Files**:
    - `Dockerfile`: Defines the Docker image for the application.
    - `docker-compose.yml`: Defines how to run the application services (including GPU support).
    - `Pipfile`: Specifies Python package dependencies.
    - `Pipfile.lock`: Locks package versions for reproducible builds.
    - Your application code (e.g., `inference.py`).

## Building and Running the Application

We will use Docker Compose to manage the build and run process.

### Step 1: Clone the Repository (if applicable)

If you haven't already, clone the project repository to your local machine:

```bash
git clone <your-repository-url>
cd <your-project-directory>
```

### Step 2: Check/Generate Pipfile.lock

The `Dockerfile` uses `pipenv install --deploy`, which requires `Pipfile.lock` to be up-to-date with `Pipfile`.

**Troubleshooting `Pipfile.lock` out-of-date error:**
If, during the Docker build process (Step 3), you encounter an error similar to:

```
Your Pipfile.lock (...) is out of date. Expected: (...).
ERROR:: Aborting deploy
```

This means your `Pipfile.lock` is not synchronized with your `Pipfile`. To fix this, run the following command in your project's root directory (where `Pipfile` is located) on your **host machine**:

```bash
pipenv lock
```

This will update `Pipfile.lock`. After running this command, proceed to Step 3.

### Step 3: Build and Run with Docker Compose

Open your terminal in the root directory of the project (where `docker-compose.yml` and `Dockerfile` are located).

**To build the image and run the application (e.g., execute `inference.py`):**

```bash
docker-compose up --build
```

- `--build`: This flag tells Docker Compose to build the Docker image using the `Dockerfile`. You can omit this on subsequent runs if the `Dockerfile` or its dependencies haven't changed, and an image already exists.
- The application (defined by `CMD` in the `Dockerfile`, e.g., `python3 inference.py`) will start, and its output will be displayed in your terminal.

**To run in detached mode (in the background):**

```bash
docker-compose up --build -d
```

### Step 4: Interacting with the Application

- **Viewing Logs (if running in detached mode):**

  ```bash
  docker-compose logs -f app
  ```

  (Replace `app` with your service name if it's different in `docker-compose.yml`). Press `Ctrl+C` to stop following logs.

- **Accessing a Shell Inside the Container (for debugging):**
  If you need to explore the container's environment or run commands manually:

  1.  Ensure the container is running (e.g., using `docker-compose up -d`).
  2.  Open a shell:
      ```bash
      docker-compose exec app bash
      ```
      (Replace `app` with your service name if it's different).
  3.  Inside the container, you can navigate to `/app` (the working directory) and run Python scripts or other commands.

- **Port Mapping (if applicable):**
  If your application (`inference.py`) runs a web server (e.g., on port 8000) and you have configured port mapping in `docker-compose.yml` (e.g., `ports: - "8000:8000"`), you can access it via `http://localhost:8000` in your web browser.

### Step 5: Stopping the Application

To stop and remove the containers, networks, and (optionally, depending on `docker-compose down` flags) volumes defined by Docker Compose:

```bash
docker-compose down
```

If you want to remove the volumes as well:

```bash
docker-compose down -v
```

## Important Notes

- **PyTorch Versions & CUDA:** The `Pipfile` specifies PyTorch versions and a CUDA source (`pytorch-cu111`). Ensure these versions are valid and available from the specified PyTorch wheel index. If `pipenv install` fails during the Docker build due to version conflicts or "Could not find a version" errors, you will need to:
  1.  Consult [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) to find compatible `torch`, `torchvision`, and `torchaudio` versions for your desired CUDA version (e.g., CUDA 11.1).
  2.  Update the versions in your `Pipfile`.
  3.  Run `pipenv lock` locally to regenerate `Pipfile.lock`.
  4.  Re-run `docker-compose up --build`.
- **GPU Usage:** The `docker-compose.yml` is configured to attempt GPU access using NVIDIA. This requires the prerequisites mentioned above (NVIDIA drivers and NVIDIA Container Toolkit on the host). If GPUs are not available or not configured correctly, PyTorch will typically fall back to CPU mode.
- **Development Mode Volume Mount:** The `docker-compose.yml` includes `volumes: - .:/app`. This mounts your local project directory into the container. Code changes made locally will be reflected inside the container, which is useful for development. For production, you might remove this volume mount to rely solely on the code baked into the image.

## Further Actions

- **Cleaning up Docker Resources:**
  - To remove unused Docker images: `docker image prune`
  - To remove unused Docker volumes: `docker volume prune`
  - To remove unused Docker networks: `docker network prune`
  - To remove all unused Docker resources (images, containers, volumes, networks): `docker system prune -a` (Use with caution!)
