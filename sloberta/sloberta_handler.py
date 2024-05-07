import torch
import logging
import transformers
import os
import json


from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoTokenizer, CamembertForQuestionAnswering

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Transformers version %s", transformers.__version__)


class SloBERTaHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        self.manifest = context.manifest

        logger.info(f"System Properties: {properties}")
        logger.info(f"Manifest: {self.manifest}")

        model_dir = properties.get("model_dir")
        logger.info(f"Model dir: {model_dir}")

        # Use GPU if available
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Device: {self.device}")

        # Load the model
        try:
            model_file = self.manifest["model"]["modelFile"]
            model_path = os.path.join(model_dir, model_file)

            logger.debug(f"Loading model from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e

        if os.path.isfile(model_path):
            self.model = CamembertForQuestionAnswering.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        else:
            raise RuntimeError(f"Model file not found at {model_path}")

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer is None:
            raise RuntimeError(f"Tokenizer not found at {model_dir}")
        else:
            logger.info(f"Tokenizer loaded from {model_dir}")

        self.initialized = True

    def preprocess(self, data):
        """
        Tokenizes the input data using the suitable tokenizer and converts it into PyTorch tensors.

        Args:
            data (json): Input data in the form of a JSON object.
        """

        logger.debug(f"Received data: {data}")

        # Unpack the data
        if data[0].get("body") is not None:
            # Standalone Torchserve inference requests are wrapped in 'body', of which value
            # is a bytearray. So, convert the data to a dictionary if it's a bytearray
            data = data[0]["body"]

            if isinstance(data, bytearray):
                data = json.loads(data.decode())

        # For example in case of KServe requests, the data is already a dictionary
        data = data[0]["data"]

        text = data["text"]
        question = data["question"]

        logger.debug(f"Received text: {text}")
        logger.debug(f"Received question: {question}")

        # Tokenize the input
        tokenized_data = self.tokenizer(question, text, return_tensors="pt")

        logger.debug(f"Tokenization complete")

        return tokenized_data

    def inference(self, inputs):
        """
        Generates predictions using the model.

        Args:
            inputs (dict): Tokenized input data in the form of a dictionary.
        """

        # Generate predictions
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[
            0, answer_start_index : answer_end_index + 1
        ]

        return predict_answer_tokens

    def postprocess(self, output):
        """
        Decodes the model output to generate the final answer.

        Args:
            inference_output (torch.Tensor): Output tensor from the model.
        """

        answer = self.tokenizer.decode(output, skip_special_tokens=True)

        logger.debug(f"Answer: {answer}")

        return [answer]

    def get_insights(self, **kwargs):
        """
        Functionality to get the explanations.
        Called from the explain_handle method
        """
        pass
