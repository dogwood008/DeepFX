FROM jupyter/datascience-notebook

USER root

# Install Java
RUN echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu precise main\ndeb-src http://ppa.launchpad.net/webupd8team/java/ubuntu precise main" >> /etc/apt/sources.list.d/java.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys EEA14886
RUN apt-get update
RUN echo debconf shared/accepted-oracle-license-v1-1 select true | \
  debconf-set-selections
RUN echo debconf shared/accepted-oracle-license-v1-1 seen true | \
  debconf-set-selections
RUN apt-get install -y oracle-java8-installer
RUN export JAVA_HOME

# Install bazel
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
        tee /etc/apt/sources.list.d/bazel.list &&\
        curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
RUN apt-get update && apt-get -y install bazel

# Install python & screen, htop
RUN apt-get -y install python-pip python-dev screen htop
RUN cd /tmp && wget 'http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz' &&\
        tar -xzvf ta-lib-0.4.0-src.tar.gz &&\
        cd ta-lib &&\
        ./configure --prefix=/usr &&\
        make &&\
        make install

# Compile TensorFlow
RUN cd /tmp &&\
        git clone https://github.com/tensorflow/tensorflow && \
        cd tensorflow &&\
        git checkout master 
ENV PYTHON_BIN_PATH /opt/conda/bin/python
ENV TF_NEED_JEMALLOC 1
ENV CC_OPT_FLAGS -march=native
ENV TF_NEED_GCP 0
ENV TF_NEED_HDFS 0
ENV TF_ENABLE_XLA 0
ENV TF_NEED_OPENCL 0
ENV TF_NEED_CUDA 0
ENV GCC_HOST_COMPILER_PATH /usr/bin/gcc
ENV TF_CUDA_VERSION 8.0
ENV CUDA_TOOLKIT_PATH /usr/local/cuda
ENV TF_CUDNN_VERSION 5
ENV CUDNN_INSTALL_PATH /usr/local/cuda
ENV TF_CUDA_COMPUTE_CAPABILITIES 3.0
ENV USE_DEFAULT_PYTHON_LIB_PATH 1

RUN cd /tmp/tensorflow &&\
        ./configure
RUN cd /tmp/tensorflow &&\
        bazel build -c opt --copt=-mavx \
        #bazel build -c opt --copt=-mavx --copt=-mfma \
        --copt=-mfpmath=both --copt=-msse4.2 -k //tensorflow/tools/pip_package:build_pip_package
        #--copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k //tensorflow/tools/pip_package:build_pip_package
RUN cd /tmp/tensorflow &&\
        bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow
RUN cd /tmp/tensorflow &&\
        chown -R jovyan .

# Upgrade pip
USER jovyan
RUN pip install --upgrade pip

# Install TensorFlow
USER root
RUN pip install /tmp/tensorflow/tensorflow-*-cp35-cp35m-linux_x86_64.whl

# Install pip packages
USER jovyan
RUN pip install backtrader scipy xgboost TA-Lib pandas gym numpy pandas keras sklearn gym
RUN pip install git+https://github.com/matthiasplappert/keras-rl.git
RUN echo "#!/bin/sh\nexec >/dev/tty 2>/dev/tty </dev/tty; /usr/bin/screen" > /home/jovyan/screen.sh &&\
        chmod +x /home/jovyan/screen.sh

ADD .screenrc /home/jovyan/
ADD jupyter_notebook_config.py /home/jovyan/.jupyter/
CMD start-notebook.sh
