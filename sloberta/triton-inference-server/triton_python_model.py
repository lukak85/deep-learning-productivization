import logging
import json

import torch
import torch.nn as nn
import transformers

from transformers import AutoTokenizer, CamembertForQuestionAnswering

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Transformers version %s", transformers.__version__)


class TritonPythonModel(nn.Module):
    def __init__(self, model):
        self.model = model

    def forward(self, inputs):

        inputs = tokenizer(question, text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs.to(device))


print(outputs)

answer_start_index = outputs["start_logits"].argmax()
answer_end_index = outputs["end_logits"].argmax()
