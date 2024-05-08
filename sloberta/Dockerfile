# Base image with GPU support
FROM pytorch/torchserve:0.10.0-gpu

# install dependencies
RUN pip install transformers

# Copy model archive
COPY pv/model-store /home/model-server/model-store/
COPY pv/config /home/model-server/config/

# run Torchserve upon running the container
CMD ["torchserve --start --model-store /home/model-server/model-store --models sloberta=sloberta.mar --ts-config /home/model-server/config/config.properties"]
