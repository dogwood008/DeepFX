#!/bin/sh
docker run -it -p 8888:8888 -p 6006:6006 -v `pwd`:/home/jovyan/work deepfx/tf_compiled_from_source start-notebook.sh --NotebookApp.password=$JUPYTER_PASSWORD
