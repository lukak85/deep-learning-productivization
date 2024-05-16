from seldon_core.seldon_client import SeldonClient

sc = SeldonClient(
    deployment_name="sloberta",
    namespace="seldon",
    gateway_endpoint="localhost:8003",
    gateway="istio",
)

r = sc.predict(transport="rest")

sc = SeldonClient(..., client_return_type="dict")
