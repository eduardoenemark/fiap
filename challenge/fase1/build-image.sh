#!/bin/bash
set -e

IMAGE_NAME="fiap-challenge-fase1-9iadt-rm370509"
IMAGE_VERSION="1.0"
CREATED_DATETIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_VERSION}"

docker build \
  --build-arg CREATED_DATETIME="${CREATED_DATETIME}" \
  -t "${IMAGE_NAME}:${IMAGE_VERSION}" \
  -t "${IMAGE_NAME}:latest" \
  .

echo "Build completed successfully: ${IMAGE_NAME}:${IMAGE_VERSION}"

