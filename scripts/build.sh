#!/bin/bash
set -e

cd "$(dirname $0)"/.. || exit 1
docker build . -t turkish_delight_nlp_api
docker tag turkish_delight_nlp_api:latest halecakir/turkish_delight_nlp_api
