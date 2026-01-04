from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.prompt.prompt_registry import PromptRegistry

@dataclass(frozen=True)
class PromptConfig:
    policy: Dict[str, Dict[int, str]]
    mode: str = "train"
    templates_dir: Optional[Union[str, Path]] = None
    verbose: bool = False

class PromptBuilder:
    def __init__(self, config: PromptConfig):
        self.cfg = config
        self.registry = PromptRegistry(
            templates_dir=self.cfg.templates_dir,
            verbose=self.cfg.verbose
        )

    def build_message(self, row: Dict[str, Any]) -> Dict[str, Any]:
        system_message = self._get_system_message(row)
        user_message = self._get_user_message(row)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        result = {"id": row.get("id"), "messages": messages}

        if self.cfg.mode == "train":
            assistant_message = self._get_assistant_message(row)
            messages.append({"role": "assistant", "content": assistant_message})
            result["label"] = int(row["answer"])
            return result
        
        return result
        
    def _get_system_message(self, row):
        choices_len = int(row["choices_len"])
        version = self.cfg.policy["system"][choices_len]
        key = f"system/{choices_len}_{version}.txt"

        template = self.registry.templates[key]
        return template

    def _get_user_message(self, row):
        choices_len = int(row["choices_len"])
        version = self.cfg.policy["user"][choices_len]

        paragraph = row["paragraph"]
        question = row["question"]
        choices = row["choices"]
        choices_str = self._format_choices(choices)

        q_plus = row.get("question_plus", None)
        has_plus = q_plus is not None and str(q_plus).strip().lower() not in ("", "nan", "none")
        p_type = "question" if has_plus else "noquestion"
        key = f"user/{choices_len}_{p_type}_{version}.txt"

        template = self.registry.templates[key]

        if has_plus:
            return template.format(
                paragraph=paragraph,
                question_plus=q_plus,
                question=question,
                choices=choices_str,
            )
        return template.format(
            paragraph=paragraph,
            question=question,
            choices=choices_str,
        )

    def _get_assistant_message(self, row):
        return str(row['answer'])

    def _format_choices(self, choices: Any) -> str:
        if isinstance(choices, list):
            return "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        return str(choices)