"""Skills: named bundles of tools (engine tools + MCP tools + memory ops).

A skill is a YAML-described bundle that pins a tool set plus optional
system-prompt hints::

    name: research
    description: "Deep web research with persistent notes"
    system_hint: "You have web access; save findings to memory."
    tools:
      - web_fetch
      - memory_write
      - memory_search
    skills_refs:          # compose other skills
      - notes

Bundles are stored under ``~/.nexinfer/skills/`` and shipped with a set
of defaults (``research``, ``coding``, ``memory``, ``default``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Skill:
    name: str
    description: str = ""
    system_hint: str = ""
    tools: list[str] = field(default_factory=list)
    skills_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "system_hint": self.system_hint,
            "tools": self.tools,
            "skills_refs": self.skills_refs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Skill":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            system_hint=d.get("system_hint", ""),
            tools=d.get("tools", []),
            skills_refs=d.get("skills_refs", []),
        )


DEFAULT_SKILLS_DIR = os.path.expanduser("~/.nexinfer/skills")

DEFAULT_SKILLS: dict[str, Skill] = {
    "default": Skill(
        name="default",
        description="Standard chat with no extra tools",
        tools=[],
    ),
    "research": Skill(
        name="research",
        description="Deep web research with persistent notes",
        system_hint="You have internet access via web_fetch. Save key findings to memory so later agents can build on them.",
        tools=["web_fetch", "memory_write", "memory_search"],
    ),
    "coding": Skill(
        name="coding",
        description="Coding assistant with file tools",
        system_hint="Use provided file tools and remember completed work in memory.",
        tools=["memory_write"],
    ),
    "memory": Skill(
        name="memory",
        description="Agent with full memory fabric access",
        tools=["memory_write", "memory_search", "memory_read", "memory_branch"],
    ),
}


class SkillsRegistry:
    def __init__(self, skills_dir: str = DEFAULT_SKILLS_DIR) -> None:
        self.skills_dir = skills_dir
        self._skills: dict[str, Skill] = dict(DEFAULT_SKILLS)
        self._load_user_skills()

    def _load_user_skills(self) -> None:
        if not os.path.isdir(self.skills_dir):
            return
        for fname in os.listdir(self.skills_dir):
            if fname.endswith((".yml", ".yaml")):
                path = os.path.join(self.skills_dir, fname)
                try:
                    with open(path) as f:
                        d = yaml.safe_load(f)
                    if isinstance(d, dict) and "name" in d:
                        self._skills[d["name"]] = Skill.from_dict(d)
                except Exception:  # noqa: BLE001
                    pass

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def save(self, skill: Skill) -> str:
        os.makedirs(self.skills_dir, exist_ok=True)
        path = os.path.join(self.skills_dir, f"{skill.name}.yml")
        with open(path, "w") as f:
            yaml.safe_dump(skill.to_dict(), f, sort_keys=False)
        self._skills[skill.name] = skill
        return path

    def resolve_tools(self, skill_name: str | None) -> tuple[Skill, list[str]]:
        """Return (skill, fully expanded tool list) with refs flattened."""
        if not skill_name:
            skill_name = "default"
        skill = self.get(skill_name) or self._skills["default"]
        seen: set[str] = set()
        tools: list[str] = []

        def _walk(s: Skill) -> None:
            for ref in s.skills_refs:
                base = self.get(ref)
                if base and base.name not in seen:
                    seen.add(base.name)
                    _walk(base)
            for t in s.tools:
                if t not in tools:
                    tools.append(t)

        seen.add(skill.name)
        _walk(skill)
        return skill, tools
