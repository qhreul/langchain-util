import os
from dotenv import load_dotenv

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings

from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig, PromptTypeEnum

load_dotenv()


def _load_ollama_llm_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a Langchain object configured for Ollama LLMs
    """
    llm = OllamaLLM(
        base_url=os.getenv('OLLAMA_ENDPOINT'),
        model=config.model_name,
        temperature=config.parameters.temperature,
        callbacks=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return llm


def _load_ollama_chat_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a Langchain chat object configured for Ollama LLMs
    """
    llm = ChatOllama(
        base_url=os.getenv('OLLAMA_ENDPOINT'),
        model=config.model_name,
        temperature=config.parameters.temperature,
        callbacks=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return llm


def _load_ollama_embeddings_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the embeddings execution
    :return: a Langchain object configured for Ollama LLMs
    """
    llm = OllamaEmbeddings(
        base_url=os.getenv('OLLAMA_ENDPOINT'),
        model=config.model_name
    )
    return llm


def load_ollama_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a OpenAI object configured for a local LLM
    """
    if config.type == PromptTypeEnum.chat:
        return _load_ollama_chat_from_prompt_config(config)
    elif config.type == PromptTypeEnum.completion:
        return _load_ollama_llm_from_prompt_config(config)
    elif config.type == PromptTypeEnum.embedding:
        return _load_ollama_embeddings_from_prompt_config(config)
