source ./setup-env-vars.sh

echo "****************** Explaining ******************"
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:explain -d @./examples/${EXAMPLE}.json

echo