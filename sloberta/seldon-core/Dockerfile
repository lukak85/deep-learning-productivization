FROM python:3.11-slim
WORKDIR /app

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Port for GRPC
EXPOSE 5000
# Port for REST
EXPOSE 9000

# Define environment variables
ENV MODEL_NAME MyModel
ENV SERVICE_TYPE MODEL

# Changing folder to default user
RUN chown -R 8888 /app

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE --debug