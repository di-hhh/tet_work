from __future__ import annotations

import os

from omegaconf import OmegaConf, open_dict


def initialize_config(config):
    try:
        with open_dict(config):
            config.slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
            config.slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    except KeyError:
        pass


def conditional_resolver(condition, if_true: str, if_false: str):
    if condition:
        return if_true
    else:
        return if_false


def shortener(input_string: str | None = None, length=3, show_config_stack=False):
    if input_string is None:
        return "job_type"
    output_parts = []

    for part in _split_top_level_overrides(input_string):
        key, separator, value = part.partition("=")
        if not separator:
            raise ValueError(f"Invalid Hydra override without '=': {part!r}")
        modified_key = ""

        key_parts = key.split(".")
        if not show_config_stack:
            key_parts = key_parts[-1:]
        for key_part in key_parts:
            for word in key_part.split("_"):
                modified_key += word[:length] + "_"
            modified_key = modified_key[:-1] + "."
        modified_key = modified_key[:-1]

        output_parts.append(f"{modified_key}={value}")

    return ",".join(output_parts)


def _split_top_level_overrides(input_string: str) -> list[str]:
    """Split Hydra's override dirname without splitting list/dict values."""
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    matching = {"]": "[", "}": "{", ")": "("}
    stack: list[str] = []

    for index, character in enumerate(input_string):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
            continue
        if character in "[{(":
            stack.append(character)
            depth += 1
            continue
        if character in matching:
            if stack and stack[-1] == matching[character]:
                stack.pop()
                depth -= 1
            continue
        if character == "," and depth == 0:
            parts.append(input_string[start:index])
            start = index + 1

    parts.append(input_string[start:])
    return [part for part in parts if part]


def load_omega_conf_resolvers():
    OmegaConf.register_new_resolver("sub_dir_shortener", shortener)
    OmegaConf.register_new_resolver("format", lambda inpt, formatter: formatter.format(inpt))
    OmegaConf.register_new_resolver("conditional_resolver", conditional_resolver)
