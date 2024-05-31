FROM python:latest

WORKDIR /project

COPY NNPC.py /project

COPY Housing_v2.csv /project

COPY requirements.txt /project

RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD ["python3", "NNPC.py"]