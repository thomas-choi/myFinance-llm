FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04

# If the host is 24.0 and CUDA should be upgraded to 12.6+.

RUN apt-get update && apt-get install -y python3.11 python3-pip python3.11-dev python3.11-venv git sudo

# Install uv package manager
RUN pip install uv

WORKDIR /app

# Create a non-root user with sudo privileges
# Build args to match host user
ARG UID=1000
ARG GID=1000
ARG USERNAME=developer

RUN groupadd -g ${GID} ${USERNAME} || true && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up the application directory with proper permissions
RUN chown -R ${UID}:${GID} /app

# Copy requirements as root first, then change ownership
COPY requirements.txt .
RUN chown ${UID}:${GID} requirements.txt

USER ${USERNAME}

# Install Python dependencies in user's home venv
ENV VIRTUAL_ENV=/app/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Clone and install timesfm from repository with torch support using uv
RUN cd /tmp && git clone https://github.com/google-research/timesfm.git && \
    cd /tmp/timesfm && \
    uv pip install -e .[torch]

    # Clone and install lag--llama from repository with torch support using uv
RUN cd /tmp && git clone https://github.com/time-series-foundation-models/lag-llama.git && \
    cd /tmp/lag-llama && \
    uv pip install -e .

COPY --chown=${UID}:${GID} src/ src/

CMD ["bash"]