FROM python:3.7.4-stretch

ARG DASHBOARD_SAGEMAKER_MODEL
ENV DASHBOARD_SAGEMAKER_MODEL=$DASHBOARD_SAGEMAKER_MODEL

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY package/ package/
RUN pip install -r package/requirements.txt
RUN pip install -e package

COPY src/ src/
COPY .env .env

EXPOSE 80

ENTRYPOINT [ "streamlit", "run", "src/app.py", \
             "--browser.serverAddress", "0.0.0.0", \
             "--server.enableCORS", "False", \
             "--server.port", "80"]