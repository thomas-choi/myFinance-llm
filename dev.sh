#!/bin/bash

# Dev container startup script
# Usage: ./dev.sh [build|start|stop|shell|logs]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get current user's UID and GID for container
export USER_UID=$(id -u)
export USER_GID=$(id -g)
export USERNAME=$(whoami)

COMPOSE_FILE="$SCRIPT_DIR/docker-compose.dev.yml"
CONTAINER_NAME="myfinance-dev"

case "${1:-shell}" in
  build)
    echo "Building development container..."
    docker compose -f "$COMPOSE_FILE" build
    ;;
  start)
    echo "Starting development container..."
    docker compose -f "$COMPOSE_FILE" up -d
    echo "Container started. Use './dev.sh shell' to connect."
    ;;
  stop)
    echo "Stopping development container..."
    docker compose -f "$COMPOSE_FILE" down
    ;;
  shell)
    echo "Connecting to development container..."
    docker compose -f "$COMPOSE_FILE" run --rm dev
    ;;
  logs)
    echo "Showing container logs..."
    docker compose -f "$COMPOSE_FILE" logs -f
    ;;
  *)
    echo "Usage: $0 [build|start|stop|shell|logs]"
    echo ""
    echo "Commands:"
    echo "  build    - Build the development container image"
    echo "  start    - Start the container in background"
    echo "  stop     - Stop the running container"
    echo "  shell    - Start interactive shell in container (default)"
    echo "  logs     - View container logs"
    exit 1
    ;;
esac
