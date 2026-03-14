FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3.12-venv \
    openssh-client \
    psmisc \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Set python3.12 as the default 'python' command
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN apt-get update
RUN apt install -y gpg-agent wget
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt update
RUN apt install -y intel-oneapi-mkl-devel

# Load oneAPI variables automatically in interactive bash terminals
RUN printf '\n# Intel oneAPI environment\nif [ -f /opt/intel/oneapi/setvars.sh ]; then\n    . /opt/intel/oneapi/setvars.sh > /dev/null 2>&1\nfi\n' >> /etc/bash.bashrc

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# This makes it run every time the container starts
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
