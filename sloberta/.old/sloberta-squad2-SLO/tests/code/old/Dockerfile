FROM python:3.7

# Set the working directory in the container
WORKDIR /app

# Copy the model files into the container at /app
COPY . /app

# Install transformers and any other required packages
RUN pip install transformers flask torch

# Expose the port the app runs on
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
