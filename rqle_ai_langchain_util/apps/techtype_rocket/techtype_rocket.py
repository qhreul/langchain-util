from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rqle_ai_langchain_util import settings
from rqle_ai_langchain_util.llms.adapters.llm_adapters import LLMAdapter
from rqle_ai_langchain_util.llms.llm_mediator import LLMMediator

# logging configuration
from time import time
import logging
from logging.config import dictConfig
dictConfig(settings.LOG_CONFIG)
logger = logging.getLogger(__name__)

load_dotenv()


class TechTypeRocket:

    def info(self):
        """
        :return: the name of the class
        """
        return self.__class__.__name__

    def __init__(self, config_folder: str):
        # configure the LLM to be executed
        self.config_folder = config_folder
        self.parser = StrOutputParser()
        self.llm_mediator = LLMMediator(LLMAdapter.OLLAMA_AI, self.config_folder)

    def load_chain(self):
        """
        :return: a configured LLM to be executed
        """
        try:
            # configure the prompt
            prompt = PromptTemplate(template=self.llm_mediator.prompt_template.prompt,
                                    input_variables=['target_reading_time', 'target_audience', 'topic', 'keywords'])
            logger.debug(f'Generated prompt: {prompt}')

            # return the LLM
            chain = prompt | self.llm_mediator.model | self.parser
            return chain
        except Exception as e:
            logger.error(f'Error loading chain: {self.config_folder}\n{e}', exc_info=True)
            raise e

    def invoke_chain(self, target_reading_time: int, target_audience: str, topic: str, keywords: list[str]):
        """
        :param target_reading_time: the targeted length of the blog in terms of reading time
        :param target_audience: the target audience of the blog
        :param topic: the topic of the blog
        :param keywords: additional keywords related to the topic
        :return: the output from executing the LLM chain
        """
        try:
            # load the chain to be executed
            chain = self.load_chain()
            # execute the chain
            output = chain.invoke({'target_reading_time': target_reading_time,
                                   'target_audience': target_audience,
                                   'topic': topic,
                                   'keywords': keywords})
            logger.debug(f'Output from TechType Rocket: {output}')
            return output
        except Exception as e:
            logger.error(f'Error executing chain: {self.config_folder}\n{e}', exc_info=True)
            raise e


if __name__ == '__main__':
    start = time()
    try:
        # create an instance of the class
        tech_type_rocket = TechTypeRocket(config_folder='techtype_rocket')
        # invoke the chain
        output = tech_type_rocket.invoke_chain(target_reading_time=5,
                                               target_audience='Business Leaders',
                                               topic='Discuss how generative AI is democratizing the adoption of AI,'
                                                     'Include a discussion on how generative AI removes barriers '
                                                     'provided by discriminative AI',
                                               keywords=['generative AI', 'discriminative AI', 'AI adoption'])
    except Exception as e:
        logger.error(f'Error executing TechType Rocket\n{e}', exc_info=True)
    finally:
        # calculate elapsed time
        total_time = round((time() - start) / 60, 3)
        logger.info(f'Elapsed time to execute TechType Rocket: {total_time} min')