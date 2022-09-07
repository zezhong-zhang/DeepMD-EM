FROM dptechnology/dflow:latest

WORKDIR /data/deepmdem
ADD requirements.txt ./
RUN pip install -r requirements.txt
COPY ./ ./
RUN pip install .
