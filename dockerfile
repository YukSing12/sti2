FROM nvcr.io/nvidia/tensorrt:21.03-py3
LABEL author="Rongrui Zhan"
RUN apt-get update
RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN apt-get install openssl openssh-server tree clang cmake gdb -y
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set global.extra-index-url "https://pypi.ngc.nvidia.com"
# RUN pip install tensorrt

RUN echo "root:1" | chpasswd
RUN mkdir -p /root/.ssh && chown root.root /root && chmod 700 /root/.ssh
RUN mkdir -p /root/.ssh && chown root.root /root && chmod 700 /root/.ssh
EXPOSE 22/tcp

ENV PATH=/usr/local/cuda/bin/:$PATH

#CMD service ssh restart
WORKDIR /target

CMD ["bash"]
# docker run --gpus all --name sti2 -p 22:10003  --shm-size=1g --ulimit memlock=-1 -it --rm nvcr.io/nvidia/paddlepaddle:22.07-py3
# export PATH=/usr/local/cuda-10.1/bin/:$PATH