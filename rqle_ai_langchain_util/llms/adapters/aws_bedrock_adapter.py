# Information about the different providers supported by AWS Bedrock -
# https://us-west-2.console.aws.amazon.com/bedrock/home

# The following models only supports "chat" as the model type:
# * us.amazon.nova-lite-v1:0
# * us.anthropic.claude-3-5-sonnet-20241022-v2:0

# The following models only support "completion" as the model type:
# * cohere.command-r-plus-v1:0

# mistral models have an issue with their handling - Error occurred while generating the blog: 'NoneType' object is not subscriptable

import os
import re
from dotenv import load_dotenv

from langchain_aws.llms.bedrock import BedrockLLM
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.embeddings.bedrock import BedrockEmbeddings

from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig, PromptTypeEnum

load_dotenv()


def _get_model_kwargs(config: PromptConfig):
    """
    Amazon Bedrock has shifted to inference profile IDs / ARNs to identify different models
    :param config: the configuration for the LLM execution
    :return: the model arguments relevant for different models supported in AWS bedrock
    """
    # us.amazon.nova-* models are only supported for chat
    if re.match('^(?:[a-z]{2}\.)?amazon\.*', config.model_name):
        return {
            "maxTokens": config.parameters.max_tokens,
            "temperature": config.parameters.temperature,
            "topP": config.parameters.top_p
        }
    # Anthropic models (e.g. us.anthropic.claude-* or anthropic.claude-*) only support chat types
    elif re.match('^(?:[a-z]{2}\.)?anthropic\.*', config.model_name):
        return {
            "max_tokens": config.parameters.max_tokens,
            "temperature": config.parameters.temperature,
            "top_p": config.parameters.top_p,
            "top_k": config.parameters.top_k
        }
    # Issue with supporting cohere models - #: extraneous key [prompt] is not permitted
    # Issue doesn't seem to be related to inclusion of '#' in prompt
    #elif re.match('^(?:[a-z]{2}\.)?cohere\.*', config.model_name):
    #    return {
    #        "max_tokens": config.parameters.max_tokens,
    #        "temperature": config.parameters.temperature,
    #        "p": config.parameters.top_p
    #    }
    elif re.match('^(?:[a-z]{2}\.)?meta\.*', config.model_name):
        return {
            "max_gen_len": config.parameters.max_tokens,
            "temperature": config.parameters.temperature,
            "top_p": config.parameters.top_p
        }
    # Known Issue with some of the mistral models running on AWS Bedrock
    # * mistral.mixtral-8x7b-instruct-v0:1 - An error occurred (ValidationException) when calling the InvokeModel operation: Validation Error
    # * mistral.mistral-large-2407-v1:0 - Error occurred while generating the blog: 'NoneType' object is not subscriptable
    elif re.match('^(?:[a-z]{2}\.)?mistral\.*', config.model_name):
        return {
            "max_tokens": config.parameters.max_tokens,
            "temperature": config.parameters.temperature,
            "top_p": config.parameters.top_p
        }
    else:
        raise NotImplementedError(f'{config.model_name} is not supported')


def _load_aws_bedrock_llm_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for AWS Bedrock LLMs
    """
    # transform the model configuration in the appropriate dictionary
    model_kwargs = _get_model_kwargs(config)
    # create a new Bedrock LLM object
    llm = BedrockLLM(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"),
        region_name=os.environ.get("BWB_REGION_NAME"),
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"),
        model_id=config.model_name,
        model_kwargs= model_kwargs
    )
    return llm


def _load_aws_bedrock_chat_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain Chat object configured for AWS Bedrock LLMs
    """
    # transform the model configuration in the appropriate dictionary
    model_kwargs = _get_model_kwargs(config)
    # create a new Bedrock LLM object
    llm = ChatBedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"),
        region_name=os.environ.get("BWB_REGION_NAME"),
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"),
        model_id=config.model_name,
        model_kwargs=model_kwargs
    )
    return llm


def _load_aws_bedrock_embeddings_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain embeddings object configured for AWS Bedrock LLMs
    """
    llm = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"),
        region_name=os.environ.get("BWB_REGION_NAME"),
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"),
        model_id=config.model_name
    )
    return llm


def load_aws_bedrock_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for AWS Bedrock LLMs
    """
    if config.type == PromptTypeEnum.chat:
        return _load_aws_bedrock_chat_from_prompt_config(config)
    elif config.type == PromptTypeEnum.completion:
        return _load_aws_bedrock_llm_from_prompt_config(config)
    elif config.type == PromptTypeEnum.embedding:
        return _load_aws_bedrock_embeddings_from_prompt_config(config)
