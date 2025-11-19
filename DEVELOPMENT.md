# Development with Docker

This guide explains how to use the Docker development environment for this project.

## Setup

The Dockerfile has been configured for development work with the following features:

1. **Live File Synchronization**: All changes to files in the host repository are immediately reflected in the container
2. **Non-root User**: The container runs as the user who built it (not root), matching your host user ID and group ID
3. **Virtual Environment**: Python packages are installed in a virtual environment within the container
4. **GPU Support**: CUDA and PyTorch are configured for GPU access

## Quick Start

### Option 1: Using the dev.sh helper script (Recommended)

```bash
# Build the development image (only needed once or when dependencies change)
./dev.sh build

# Start an interactive shell in the container
./dev.sh shell

# Or: start container in background and connect to it
./dev.sh start
./dev.sh shell
./dev.sh logs  # view logs in another terminal

# Stop the container
./dev.sh stop
```

### Option 2: Direct docker compose commands

```bash
# Build with your user credentials
docker-compose -f docker-compose.dev.yml build

# Start interactive session
docker-compose -f docker-compose.dev.yml run --rm dev bash

# Or start in background
docker-compose -f docker-compose.dev.yml up -d
docker-compose -f docker-compose.dev.yml exec dev bash
docker-compose -f docker-compose.dev.yml down
```

## How It Works

### User Mapping
The container automatically creates a user with the same UID/GID as your host system. This ensures:
- Files created in the container are owned by your user on the host
- File permissions remain consistent
- No permission issues when editing files

The build uses build arguments:
- `UID`: Your user ID (automatically set from `id -u`)
- `GID`: Your group ID (automatically set from `id -g`)
- `USERNAME`: Your username (automatically set)

### Volume Mounts
The development compose file mounts:
- `.:/app` - Your entire project folder for live sync
- `/app/__pycache__` - Named volume to avoid sync of cache files
- `/app/.pytest_cache` - Named volume for pytest cache
- `/app/venv` - Named volume for virtual environment

## Development Workflow

### Making Code Changes

Simply edit files in your host editor/IDE:

```bash
# In host terminal
$EDITOR src/data_prep.py

# In container, changes are immediately available
$ python src/data_prep.py --help
```

### Installing New Dependencies

If you need to install new Python packages:

```bash
# Option 1: Update requirements.txt and rebuild
echo "new-package==1.0.0" >> requirements.txt
./dev.sh build

# Option 2: Install directly in container and update requirements later
pip install new-package
pip freeze > requirements.txt
# Then rebuild to make it permanent
```

### Running Tests

```bash
./dev.sh shell
python src/test.py
```

### Running Training/Inference

```bash
./dev.sh shell
python src/finetune.py --model_name google/timesfm-1.0-200m --data data/prepared_data.csv
```

## GPU Support

GPU support is enabled automatically if you have NVIDIA Docker runtime installed.

Verify GPU access in the container:
```bash
./dev.sh shell
nvidia-smi
```

## Troubleshooting

### Permission Denied Errors
If you get permission errors, the container user may not have been created with the correct UID/GID:
```bash
# Rebuild with correct user mapping
./dev.sh build
./dev.sh shell
id  # Verify UID/GID match your host `id` output
```

### Files Not Syncing
- Check that you mounted the correct volume: `docker inspect myfinance-dev`
- Ensure you're editing files in the mounted directory
- For some editors, you may need to disable file caching

### CUDA/GPU Not Available
```bash
# Check Docker GPU setup
docker run --rm --gpus all ubuntu nvidia-smi

# Rebuild and ensure nvidia runtime is configured
./dev.sh build
```

## Advanced: Custom User/Project Structure

If you need custom build arguments:

```bash
docker compose -f docker-compose.dev.yml build --build-arg USER_UID=2000 --build-arg USERNAME=custom_user
```

Or modify `docker-compose.dev.yml` directly:
```yaml
build:
  args:
    USER_UID: 2000
    USER_GID: 2000
    USERNAME: custom_user
```
