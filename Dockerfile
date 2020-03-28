#FROM nvcr.io/nvidia/tensorflow:19.07-py3
FROM tatsu.ism.lab/tensorflow/tensorflow:1.15.0-gpu-py3-jupyter
#FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

ARG PYPI_HOST=perry.ism.lab
ARG PIP_INDEX_URL=http://${PYPI_HOST}/pypi/simple/
ARG CODE_DIR=/code

ARG DATA_DIR=/data
ARG RESULTS_DIRECTORY=/results

RUN echo "deb http://perry.ism.lab/ubuntu bionic main multiverse universe restricted" > /etc/apt/sources.list

RUN apt-get update && apt-get install -y dcm2niix python-gdcm vim

COPY ./requirements.txt /
RUN pip3 install -r /requirements.txt --index-url=${PIP_INDEX_URL} --trusted-host=${PYPI_HOST}

RUN mkdir -p ${CODE_DIR}

ENV CODE_DIR ${CODE_DIR}
ENV PYTHONPATH ${CODE_DIR}

ENV DATA_PATH "${DATA_DIR}"
ENV MODEL_CONF ${MODEL_CONFIG}
ENV EXP_NAME ${EXPERIMENT_NAME}
ENV SAVE_DIR ${RESULTS_DIRECT}
WORKDIR /code
EXPOSE 8888

