# syntax = docker/dockerfile:1.4
ARG MAMBA_VERSION=1.0.0

# https://github.com/mamba-org/micromamba-docker
FROM docker.io/mambaorg/micromamba:${MAMBA_VERSION} as app

ENV SHELL=/bin/bash \
    LANG=C.UTF-8  \
    LC_ALL=C.UTF-8
    # Need this environment variable to build pytorch with CUDA on non-GPU machine:

# chown for default 'mambauser' permissions
COPY --link --chown=1000:1000 environment.yml* /tmp/env.yml

RUN --mount=type=cache,target=/opt/conda/pkgs <<eot
    micromamba install -y -n base -f /tmp/env.yml
    micromamba clean --all --yes
eot

# Copy all repository files
COPY --link --chown=1000:1000 ./ /tmp

# do we need to add command in this docker image?
RUN micromamba clean --all -y