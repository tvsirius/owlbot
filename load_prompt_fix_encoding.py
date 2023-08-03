import json
import logging
from pathlib import Path
from typing import Union

import yaml

from langchain.output_parsers.regex import RegexParser
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseLLMOutputParser, BasePromptTemplate, StrOutputParser
from langchain.utilities.loading import try_load_from_hub


from langchain.prompts.loading import load_prompt_from_config


#
#
#  this is fix file for loading with encoding='utf-8'
#
#
#

def load_prompt_utf(path: Union[str, Path]) -> BasePromptTemplate:
    """Unified method for loading a prompt from LangChainHub or local fs."""
    if hub_result := try_load_from_hub(
        path, _load_prompt_from_file, "prompts", {"py", "json", "yaml"}
    ):
        return hub_result
    else:
        return _load_prompt_from_file(path)


def _load_prompt_from_file(file: Union[str, Path]) -> BasePromptTemplate:
    """Load prompt from file."""
    # Convert file to a Path object.
    if isinstance(file, str):
        file_path = Path(file)
    else:
        file_path = file
    # Load from either json or yaml.
    if file_path.suffix == ".json":
        with open(file_path, encoding='utf-8') as f:
            config = json.load(f)
    elif file_path.suffix == ".yaml":
        with open(file_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Got unsupported file type {file_path.suffix}")
    # Load the prompt from the config now.
    return load_prompt_from_config(config)