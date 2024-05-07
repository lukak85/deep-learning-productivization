from abc import ABC
import json
import logging
import os

import torch
from transformers import AutoTokenizer, CamembertForQuestionAnswering

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersQAHandler(BaseHandler, ABC):
    """
    The handler class for the transformers-based question answering model.
    """

    def __init__(self):
        super(TransformersQAHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        """
        Initializes the model and tokenizer.
        """
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("sloberta-squad2-SLO")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = CamembertForQuestionAnswering.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.debug(f"Model loaded from {model_dir}")

        self.initialized = True

    def preprocess(self, data):
        """
        Preprocesses the input data for inference.
        """
        question, text = data[0].get("question"), data[0].get("text")
        inputs = self.tokenizer(question, text, return_tensors="pt")
        return inputs

    def inference(self, inputs):
        """
        Predicts the answer using the model.
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def postprocess(self, inference_output):
        """
        Processes the inference output to extract the answer.
        """
        answer_start_index = inference_output.start_logits.argmax()
        answer_end_index = inference_output.end_logits.argmax()
        predict_answer_tokens = inference_output.input_ids[
            0, answer_start_index : answer_end_index + 1
        ]
        answer = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        return [answer]


_service = TransformersQAHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
