ARG SCIKIT_LEARN_IMAGE
FROM $SCIKIT_LEARN_IMAGE

COPY requirements.txt /requirements.txt
RUN pip install --no-cache -r /requirements.txt && \
    rm /requirements.txt
    
# additional container dependencies can be added here