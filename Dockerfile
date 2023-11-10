FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workdir

COPY src/ src/
COPY requirements.txt .
RUN apt-get update -y
RUN apt-get install -y python3.10 python3-pip git
RUN pip install -r requirements.txt