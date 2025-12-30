from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict


class PromptRegistry:
    def __init__(
        self,
        templates_dir: Optional[str | Path] = None,
        verbose: bool = False
    ):
        if templates_dir is None:
            self.templates_dir = Path(__file__).parent / "templates"
        else:
            self.templates_dir = Path(templates_dir)
        
        self.templates: Dict[str, str] = {}
        self.verbose = verbose

        self._load_templates()

    def _load_templates(self) -> None:
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"templates_dir 경로가 존재하지 않습니다: {self.templates_dir}")

        candidates = defaultdict(dict)

        for file_path in self.templates_dir.rglob("*.txt"):
            parsed = self._parse_filename(file_path)
            if parsed is None:
                continue
            
            role, choices_len, p_type, name = parsed

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                if self.verbose:
                    print(f"파일 읽기 실패: {file_path}")
                continue

            key = f"{role}/{file_path.name}"
            self.templates[key] = content

            if role == "user":
                base_key = (role, choices_len, name)
                candidates[base_key][p_type] = key
        
        self._validate_user_pairs(candidates)

        if self.verbose:
            system_cnt = sum(k.startswith("system/") for k in self.templates)
            user4_cnt = sum(k.startswith("user/4_") for k in self.templates)
            user5_cnt = sum(k.startswith("user/5_") for k in self.templates)
            print(f"template loading 완료: system={system_cnt}, user_4={user4_cnt}, user_5={user5_cnt}")

    def _parse_filename(self, file_path: Path) -> Optional[Tuple[str, str, str, str]]:
        """
        파일명 규칙: {role}/{choices_len}_{type}_{name}.txt
        - role: system/user
        - choices_len: 4/5
        - type: user의 경우 question/ noquestion
        - name: 이름
        """
        
        # 규칙에 맞는것만 가져오기. -> 맞지 않는경우에는? -> if verbose: 라면 print로 {file_name}은 안맞다고 하고 다음으로 넘어가는걸로.
        role = file_path.parent.name
        stem = file_path.stem
        parts = stem.split('_')

        if role == "system":
            if len(parts) == 2:
                choices_len, name = parts
                return role, choices_len, "default", name
        
        elif role == "user":
            if len(parts) == 3:
                choices_len, p_type, name = parts
                if p_type in ("question", "noquestion"):
                    return role, choices_len, p_type, name
        
        if self.verbose:
            print(f"규칙에 맞지 않는 파일명입니다. {file_path.name}")
        return None
    
    def _validate_user_pairs(self, candidates: dict) -> None:
        """
        user의 경우 'question', 'noquestion' 쌍이 모두 존재하는지 검증
        """

        for (role, choices_len, name), type_map in candidates.items():
            if role != "user":
                continue

            has_q = "question" in type_map
            has_not_q = "noquestion" in type_map
        
            if has_q and has_not_q:
                continue
            
            if self.verbose:
                print(f"user pair 누락: choices_len={choices_len}, name={name}, have={list(type_map.keys())}")

            for key in type_map.values():
                self.templates.pop(key, None)