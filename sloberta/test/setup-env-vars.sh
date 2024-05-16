#!/bin/bash
EXAMPLE=example1

MODEL_NAME=sloberta
SERVICE_HOSTNAME=$(kubectl get inferenceservice torchserve -o jsonpath='{.status.url}' | cut -d "/" -f 3)

export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')

echo "Model Name: ${MODEL_NAME}"
echo "Service Hostname: ${SERVICE_HOSTNAME}"
echo "Ingress Host: ${INGRESS_HOST}"
echo "Ingress Port: ${INGRESS_PORT}"