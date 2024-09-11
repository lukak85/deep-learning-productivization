import numpy as np
from transformers import pipeline
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.generator = pipeline("question-answering", model="./model_repository/sloberta/1/model-dir")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Decode the Byte Tensor into Text
            text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()[0].decode("utf-8")
            question = pb_utils.get_input_tensor_by_name(request, "question").as_numpy()[0].decode("utf-8")

            # Call the Model pipeline
            answer = self.generator(question=question, context=text)

            # Encode the text to byte tensor to send back
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("answer", np.array(answer, dtype=object))]
            )
        responses.append(inference_response)
        return responses

    def finalize(self, args):
        self.generator = None