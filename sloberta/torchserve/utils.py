import json
import random


def get_example(examples_file="./../dataset/squad2-slo-mt-dev.json", idx=-1):
    """
    Get example from json file
    :param examples_file: path to json file
    :param idx: index of the example; if -1, return random example
    :return: context, question, answers
    """

    # Read json file
    with open(examples_file, "r") as f:
        examples = json.load(f)["data"]

    if idx == -1:
        idx = random.randint(0, len(examples) - 1)

    example = examples[idx]

    context = example["context"]
    question = example["question"]
    answers = example["answers"]

    return context, question, answers
