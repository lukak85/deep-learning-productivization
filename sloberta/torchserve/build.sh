torch-model-archiver --model-name sloberta --version 1.0 \
    --model-file ./../model-dir/pytorch_model.bin \
    --handler sloberta_handler.py \
    --extra-files ./../model-dir/ \
    --export-path ./../pv/model-store/ \
    --force