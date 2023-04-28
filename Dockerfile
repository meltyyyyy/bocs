FROM python:3.8
USER root

RUN apt-get update && \
    apt-get install -y \
    vim \
    less \
    git \
    zip \
    gnuplot

RUN mkdir -p /root/bocs
COPY ./ /root/bocs/
WORKDIR /root/bocs
ENV PYTHONPATH=/root/bocs

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
# RUN pip install -r requirements.txt
