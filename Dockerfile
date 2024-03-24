FROM unit8/darts:latest
LABEL authors="iffi"

RUN apt update -y && \
    apt install -y python3-pip && \
    apt clean && \
    apt upgrade -y

RUN pip install --upgrade pip && \
    pip install jupyterlab && \
    pip install darts

RUN #useradd -ms /bin/bash iffi
# Set the user to use when running the container
RUN #chown -R iffi:iffi /app


RUN pip install -U kaleido

# Check if directory exists before creating it
RUN #mkdir -p /app
#USER iffi

EXPOSE 8000

CMD ["jupyter", "lab", "--ip=0.0.0.0","--port=8000", "--no-browser", "--allow-root"]
