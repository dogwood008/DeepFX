FROM dogwood008/jupyter-tensorflow-notebook-for-nvidia-cuda-8.0-cudnn5-devel-ubuntu14.04

USER root

# Install python & screen, htop
RUN apt-get update && apt-get -y install python-pip python-dev screen htop curl
RUN cd /tmp && wget 'http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz' &&\
        tar -xzvf ta-lib-0.4.0-src.tar.gz &&\
        cd ta-lib &&\
        ./configure --prefix=/usr &&\
        make &&\
        make install

RUN cd /tmp && wget https://github.com/direnv/direnv/releases/download/v2.13.1/direnv.linux-amd64 && \
			chmod +x direnv.linux-amd64 && mv direnv.linux-amd64 /usr/local/bin/direnv

# Upgrade pip
USER jovyan
RUN pip install --upgrade pip

# Install pip packages
USER jovyan
RUN pip install backtrader scipy xgboost TA-Lib pandas gym numpy pandas keras sklearn gym google-api-python-client jupyter_contrib_nbextensions jupyterthemes google-api-python-client google-cloud-logging crcmod tensorflow-gpu

# Install keras-rl
RUN pip install git+https://github.com/matthiasplappert/keras-rl.git

RUN echo "#!/bin/sh\nexec >/dev/tty 2>/dev/tty </dev/tty; /usr/bin/screen" > /home/jovyan/screen.sh &&\
        chmod +x /home/jovyan/screen.sh

# Setup extensions
RUN jupyter contrib nbextension install --user --skip-running-check && \
      jt -t onedork -vim && \
			mkdir -p $(jupyter --data-dir)/nbextensions && \
			cd $(jupyter --data-dir)/nbextensions && \
			git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding || \
			jupyter nbextension enable vim_binding/vim_binding

ADD .screenrc /home/jovyan/
ADD jupyter_notebook_config.py /home/jovyan/.jupyter/
ENV TZ JST-9
ENV GOOGLE_CLOUD_STORAGE_BUCKET $GOOGLE_CLOUD_STORAGE_BUCKET
ENV GOOGLE_SERVICE_ACCOUNT_JSON_PATH $GOOGLE_SERVICE_ACCOUNT_JSON_PATH
ENV JUPYTER_PASSWORD_HASH $JUPYTER_PASSWORD_HASH
CMD start-notebook.sh
