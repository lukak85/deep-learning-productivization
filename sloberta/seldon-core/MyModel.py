import logging
import os

import torch
import transformers
from transformers import AutoTokenizer, CamembertForQuestionAnswering


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Transformers version %s", transformers.__version__)

model_dir = "./model-dir"


class MyModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        # torch.multiprocessing.set_start_method("spawn")

        # Use GPU if available
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Model dir: {model_dir}")
        self.device = torch.device("cpu")  # TODO: Fix GPU support
        logger.info(f"Device: {self.device}")

        # List all the files and subdirectories in the current directory
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"List of files and directories in the current directory:")
        for name in os.listdir("."):
            logger.info(name)

        # Load the model
        self.model = CamembertForQuestionAnswering.from_pretrained(model_dir)
        if self.model is None:
            raise RuntimeError(f"Model file not found at {model_dir}")
        else:
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded from {model_dir}")

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer is None:
            raise RuntimeError(f"Tokenizer not found at {model_dir}")
        else:
            logger.info(f"Tokenizer loaded from {model_dir}")

        self.initialized = True

    def predict(self, X, features_names=None):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        logger.debug(f"Received data: {X}")
        inputs = self.tokenizer(X[0]["question"], X[0]["text"], return_tensors="pt")
        logger.debug(f"Inputs tokenized: {inputs}")

        # Generate predictions
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(**inputs)
            logger.debug(f"Outputs: {outputs}")

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        logger.debug(f"Answer start index: {answer_start_index}")
        logger.debug(f"Answer end index: {answer_end_index}")

        predict_answer_tokens = inputs.input_ids[
            0, answer_start_index : answer_end_index + 1
        ]
        logger.debug(f"Predicted answer tokens: {predict_answer_tokens}")

        answer = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        logger.debug(f"Answer: {answer}")

        return answer
