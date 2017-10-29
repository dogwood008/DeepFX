FROM jupyter/datascience-notebook

USER root

# Install python & screen, htop
RUN apt-get update && apt-get -y install python-pip python-dev screen htop
RUN cd /tmp && wget 'http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz' &&\
        tar -xzvf ta-lib-0.4.0-src.tar.gz &&\
        cd ta-lib &&\
        ./configure --prefix=/usr &&\
        make &&\
        make install

# Upgrade pip
USER jovyan
RUN pip install --upgrade pip

# Install pip packages
USER jovyan
RUN pip install backtrader scipy xgboost TA-Lib pandas gym numpy pandas keras sklearn gym tensorflow google-api-python-client jupyter_contrib_nbextensions jupyterthemes
RUN pip install git+https://github.com/matthiasplappert/keras-rl.git
RUN echo "#!/bin/sh\nexec >/dev/tty 2>/dev/tty </dev/tty; /usr/bin/screen" > /home/jovyan/screen.sh &&\
        chmod +x /home/jovyan/screen.sh

RUN jupyter contrib nbextension install --user --skip-running-check && \
      jt -t onedork -vim && \
			mkdir -p $(jupyter --data-dir)/nbextensions && \
			cd $(jupyter --data-dir)/nbextensions && \
			git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding && \
			jupyter nbextension enable vim_binding/vim_binding

ADD .screenrc /home/jovyan/
ADD jupyter_notebook_config.py /home/jovyan/.jupyter/
CMD start-notebook.sh
