FROM amazoncorretto:8

RUN yum -y update
RUN yum -y install yum-utils
RUN yum -y groupinstall development

RUN yum list python3*
RUN yum -y install python3 python3-dev python3-pip python3-virtualenv python-dev 

RUN python -V
RUN python3 -V

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip
RUN pip3 install 'numpy==1.17.5'
RUN pip3 install 'wheel==0.36.2'

RUN python3 -c "import numpy as np"

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
