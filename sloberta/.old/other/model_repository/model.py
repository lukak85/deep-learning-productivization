from transformers import ViTFeatureExtractor, ViTModel
import torch
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def __init__(self):
        self.feature_extractor = None
        self.model = None

    def initialize(self, args):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "image")
            input_image = np.squeeze(inp.as_numpy()).transpose((2, 0, 1))
            inputs = self.feature_extractor(images=input_image, return_tensors="pt")

            # Model inference (turn off gradients to save memory)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Sending results
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "label",
                    outputs.last_hidden_state.numpy()
                )
            ])
            responses.append(inference_response)
        return responses

    # This method is called when the pipeline is completed. It is not required.
    def finalize(self):
        pass
