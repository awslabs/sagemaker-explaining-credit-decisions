FROM python:3.7.4-slim-stretch

ARG DASHBOARD_SAGEMAKER_MODEL
ENV DASHBOARD_SAGEMAKER_MODEL=$DASHBOARD_SAGEMAKER_MODEL

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
COPY src/ src/

RUN pip install -r requirements.txt

EXPOSE 80

ENTRYPOINT [ "streamlit", "run", "src/app.py", \
             "--browser.serverAddress", "0.0.0.0", \
             "--server.enableCORS", "False", \
             "--server.port", "80"]