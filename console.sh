#!/bin/sh
nvidia-docker run -it -p 8888:8888 --volume "$PWD:/home/jovyan" -t dogwood008/deepfx bash
