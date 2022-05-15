#!/bin/bash
set -e

docker pull halecakir/turkish_delight_nlp_api
docker run -p 8001:8000 -d --name turkish_delight_nlp_api --rm halecakir/turkish_delight_nlp_api
