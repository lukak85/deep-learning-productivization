source ./setup-env-vars.sh

echo "****************** Predicting ******************"
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d @./examples/${EXAMPLE}.json

echo