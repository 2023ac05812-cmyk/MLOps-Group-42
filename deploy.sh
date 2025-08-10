#!/usr/bin/env bash
IMAGE="nupur15/mlops-iris:latest"
docker pull $IMAGE
docker run -d --name mlops-iris -p 8000:8000 $IMAGE
