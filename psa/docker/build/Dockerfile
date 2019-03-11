FROM onlytailei/pytorch:1.0-cuda10-tensorboardx-scipy-skimage

RUN conda install -c anaconda mxnet

RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /tmp/*

WORKDIR /home
