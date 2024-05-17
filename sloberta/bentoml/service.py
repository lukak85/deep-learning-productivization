from __future__ import annotations
import bentoml
from transformers import pipeline


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class QuestionAnsweringService:
    def __init__(self):
        self.pipeline = pipeline("question-answering", model="../model-dir")

    @bentoml.api
    def answer(self, question: str, text: str):
        print("Question:", question)
        print("Context:", text)
        return self.pipeline(question=question, context=text)
