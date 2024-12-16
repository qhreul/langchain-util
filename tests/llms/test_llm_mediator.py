from unittest import TestCase, main
from parameterized import parameterized

from langchain_aws.llms.bedrock import BedrockLLM
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_openai import AzureOpenAI

from rqle_ai_langchain_util.llms.llm_mediator import LLMMediator
from rqle_ai_langchain_util.llms.adapters.llm_adapters import LLMAdapter
from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig, ExecutionParameters
from rqle_ai_langchain_util.settings import PROMPT_CONFIG_FOLDER


class TestLLMMediator(TestCase):

    def test_llm_mediator_init_default(self):
        mediator = LLMMediator(LLMAdapter.OLLAMA_AI)
        self.assertEqual(mediator._llm_adapter, LLMAdapter.OLLAMA_AI)
        self.assertIsNone(mediator.prompt_config)
        self.assertIsNone(mediator.prompt_template)
        self.assertIsNone(mediator.prompt_example)
        self.assertIsNone(mediator.model)

    def test_llm_mediator_init_with_config(self):
        prompt_config_test = 'techtype_rocket'

        mediator = LLMMediator(LLMAdapter.OLLAMA_AI, prompt_config_test)
        self.assertEqual(mediator.prompt_name, prompt_config_test)
        self.assertEqual(mediator.prompt_config_path, f'{PROMPT_CONFIG_FOLDER}/{prompt_config_test}')
        self.assertIsNotNone(mediator.prompt_config)
        self.assertIsNotNone(mediator.prompt_template)
        self.assertIsNotNone(mediator.model)

    def test_set_prompt_config(self):
        model_name_test = 'test_model'
        prompt_config = PromptConfig(model_name=model_name_test, parameters=ExecutionParameters())

        mediator = LLMMediator(LLMAdapter.OLLAMA_AI)
        mediator.prompt_config = prompt_config
        self.assertEqual(mediator.prompt_config, prompt_config)
        self.assertIsNotNone(mediator.model)

    def test_set_prompt_template(self):
        prompt_text = 'This is a test prompt.'

        mediator = LLMMediator(LLMAdapter.OLLAMA_AI)
        mediator.prompt_template = prompt_text
        self.assertEqual(mediator.prompt_template.prompt, prompt_text)

    @parameterized.expand([
        [LLMAdapter.AWS_BEDROCK, PromptConfig(model_name='meta.llama3-1-70b-instruct-v1', parameters=ExecutionParameters()), BedrockLLM],
        [LLMAdapter.AZURE_OPENAI, PromptConfig(model_name='', parameters=ExecutionParameters()), AzureOpenAI],
        [LLMAdapter.GOOGLE_GEMINI, PromptConfig(model_name='gemini-pro', parameters=ExecutionParameters()), GoogleGenerativeAI],
        [LLMAdapter.OLLAMA_AI, PromptConfig(model_name='', parameters=ExecutionParameters()), OllamaLLM],
    ])
    def test_load_model(self, llm_adapter: LLMAdapter, prompt_config: PromptConfig, model_class):
        mediator = LLMMediator(llm_adapter)
        mediator.prompt_config = prompt_config

        self.assertIsInstance(mediator.model, model_class)


if __name__ == '__main__':
    main()
