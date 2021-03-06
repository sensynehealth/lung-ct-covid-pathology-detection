FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

RUN apt-get update && apt-get install -y dcm2niix python-gdcm vim

COPY ./requirements.txt /
RUN pip3 install -U pip
RUN pip3 install -r /requirements.txt

WORKDIR /code

