#!/bin/sh
docker container exec -it `docker ps --format "{{.ID}}" --filter 'status=running' --filter 'ancestor=deepfx/tf_compiled_from_source'` tensorboard --log=log
