from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OptionDefinition:
    id: str
    label: str
    meaning: str


@dataclass(frozen=True)
class QuestionDefinition:
    id: str
    order: int
    topic: str
    question: str
    completion_role: str
    options: tuple[OptionDefinition, ...]


class BusinessSpec:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.version = payload["version"]
        self.question_order = list(payload["question_order"])
        self._questions: dict[str, QuestionDefinition] = {}

        for q in payload["questions"]:
            options = tuple(
                OptionDefinition(
                    id=o["id"],
                    label=o["label"],
                    meaning=o.get("meaning", o["label"]),
                )
                for o in q.get("main_options", [])
            )
            self._questions[q["id"]] = QuestionDefinition(
                id=q["id"],
                order=q["order"],
                topic=q["topic"],
                question=q["question"],
                completion_role=q.get("completion_role", "required_meaning"),
                options=options,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BusinessSpec":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    def question(self, question_id: str) -> QuestionDefinition:
        return self._questions[question_id]

    def option(self, question_id: str, option_id: str) -> OptionDefinition:
        for option in self.question(question_id).options:
            if option.id == option_id:
                return option
        raise KeyError(f"{option_id} is not valid for {question_id}")

    def match_exact_choice(self, question_id: str, text: str) -> OptionDefinition | None:
        def normalise(value: str) -> str:
            value = value.lower().replace("’", "'")
            value = re.sub(r"[–—−]", "-", value)
            return " ".join(value.strip().rstrip(".! ").split())
        normalised = normalise(text)

        for option in self.question(question_id).options:
            if normalised == option.id.lower():
                return option
            if normalised == normalise(option.label):
                return option

        return None
