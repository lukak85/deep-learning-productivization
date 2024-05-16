docker run --gpus all -m 4g --cpus 2 \
    --name sloberta-seldon \
    --rm \
    -p 5000:5000 -p 9000:9000