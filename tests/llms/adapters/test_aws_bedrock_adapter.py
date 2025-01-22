from unittest import TestCase, main
from parameterized import parameterized

from langchain_aws.llms.bedrock import BedrockLLM
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.embeddings.bedrock import BedrockEmbeddings

from rqle_ai_langchain_util.llms.adapters.aws_bedrock_adapter import load_aws_bedrock_from_prompt_config, _get_model_kwargs
from rqle_ai_langchain_util.prompts.prompt_config import ExecutionParameters, PromptConfig, PromptTypeEnum


class TestAWSBedrockAdapter(TestCase):

    @parameterized.expand([
        ('us.amazon.nova-lite-v1:0', {'maxTokens': 100, 'temperature': 0.5, 'topP': 0.5}),
        ('amazon.nova-lite-v1:0', {'maxTokens': 100, 'temperature': 0.5, 'topP': 0.5}),
        ('us.anthropic.claude-3-5-sonnet-20241022-v2:0', {'max_tokens': 100, 'temperature': 0.5, 'top_p': 0.5, 'top_k': 0}),
        ('anthropic.claude-3-5-haiku-20241022-v1:0', {'max_tokens': 100, 'temperature': 0.5, 'top_p': 0.5, 'top_k': 0}),
        ('us.meta.llama3-3-70b-instruct-v1:0', {'max_gen_len': 100, 'temperature': 0.5, 'top_p': 0.5}),
        ('meta.llama3-1-405b-instruct-v1:0', {'max_gen_len': 100, 'temperature': 0.5, 'top_p': 0.5}),
        ('mistral.mistral-large-2407-v1:0', {'max_tokens': 100, 'temperature': 0.5, 'top_p': 0.5}),
    ])
    def test_get_model_kwargs(self, provider, expected):
        config = PromptConfig(model_name=f'{provider}.test',
                              parameters=ExecutionParameters(max_tokens=100, temperature=0.5, top_p=0.5,
                                                             presence_penalty=0.5, frequency_penalty=0.5))
        kwargs = _get_model_kwargs(config)
        self.assertEqual(kwargs, expected)

    def test_get_model_kwargs_invalid_provider(self):
        config = PromptConfig(model_name='invalid.test',
                              parameters=ExecutionParameters(max_tokens=100, temperature=0.5, top_p=0.5))
        with self.assertRaises(NotImplementedError):
            _get_model_kwargs(config)

    @parameterized.expand([
        [PromptTypeEnum.chat, ChatBedrock],
        [PromptTypeEnum.completion, BedrockLLM],
        [PromptTypeEnum.embedding, BedrockEmbeddings]
    ])
    def test_load_aws_bedrock_from_prompt_config(self, llm_type: PromptTypeEnum, llm_class):
        config = PromptConfig(type=llm_type, model_name='mistral',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        result = load_aws_bedrock_from_prompt_config(config)
        self.assertIsInstance(result, llm_class)


if __name__ == '__main__':
    main()
