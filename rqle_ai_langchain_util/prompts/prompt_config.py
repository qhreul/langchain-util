from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

from rqle_ai_langchain_util.utils.file_util import read_file


class ExecutionParameters(BaseModel):
    max_tokens: int = Field(description='Max number of tokens supported by LLM', default=2000)
    temperature: float = Field(description='Temperature to control the creativity of the LLM',
                               default=0.0, ge=0.0, le=1.0)
    top_p: float = Field(description='top_p controls the range of tokens considered by the LLM',
                         default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(description='top_k limits the number of most likely tokens to consider when generating a response',
                         default=0, ge=0)
    presence_penalty: float = Field(description='Presence penalty aims at including a diverse range of tokens '
                                                'in the generated text', default=0.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(description='Frequency penalty aims at discouraging from repeating the '
                                                 'same words or phrases to frequently in the generated text',
                                     default=0.0, ge=0.0, le=1.0)


class PromptTypeEnum(str, Enum):
    completion = 'completion'
    chat = 'chat'
    embedding = 'embedding'


class PromptConfig(BaseModel):
    version: int = Field(description='Schema version', default=1)
    type: PromptTypeEnum = Field(description='Type of prompt', default=PromptTypeEnum.completion)
    model_name: str = Field(description='Name of the LL model')
    deployment_name: Optional[str] = Field(description='Name of the LL deployment', default='')
    parameters: ExecutionParameters = Field(description='Parameters for the execution of the LLM')

    @staticmethod
    def from_json(config_json_path: str) -> 'PromptConfig':
        """
        Create a PromptConfig from a JSON string
        :param config_json_path: Prompt configuration as a JSON string
        :return: a PromptConfig object
        """
        return PromptConfig.model_validate_json(read_file(config_json_path, 'config.json'))
