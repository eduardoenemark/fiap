#!/bin/bash
set -e

IMAGE_NAME="fiap-challenge-fase1-9iadt-rm370509"
IMAGE_VERSION="1.0"
CREATED_DATETIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if command -v docker &>/dev/null; then
  CONTAINER_CLI="docker"
elif command -v podman &>/dev/null; then
  CONTAINER_CLI="podman"
else
  echo "Error: neither docker nor podman was found. Please install one of them."
  exit 1
fi

echo "Using: ${CONTAINER_CLI}"
echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_VERSION}"

${CONTAINER_CLI} build \
  --build-arg CREATED_DATETIME="${CREATED_DATETIME}" \
  -t "${IMAGE_NAME}:${IMAGE_VERSION}" \
  -t "${IMAGE_NAME}:latest" \
  .

echo "Build completed successfully: ${IMAGE_NAME}:${IMAGE_VERSION}"

