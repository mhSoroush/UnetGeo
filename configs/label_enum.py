from enum import Enum


class Label(Enum):
    QUESTION = "qa.question"
    ANSWER = "qa.answer"
    HEADER = "header.header"
    OTHER = "other.other"