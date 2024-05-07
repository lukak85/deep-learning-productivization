# Base image with GPU support
FROM pytorch/torchserve:0.10.0-gpu

# install dependencies
RUN pip install transformers

# Copy model archive
COPY model_store /home/model-server/model-store/

# run Torchserve upon running the container
CMD ["torchserve --start --model-store /home/model-server/model-store --models sloberta=sloberta.mar --ts-config /home/model-server/config.properties"]
