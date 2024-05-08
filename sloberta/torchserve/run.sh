torchserve --start \
    --model-store ./../pv/model-store/ \
    --models sloberta.mar \
    --ts-config ./../pv/config/config.properties \
    --ncs