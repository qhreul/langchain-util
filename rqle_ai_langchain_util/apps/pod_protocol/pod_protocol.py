from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rqle_ai_langchain_util import settings
from rqle_ai_langchain_util.utils import video_util, file_util
from rqle_ai_langchain_util.llms.adapters.llm_adapters import LLMAdapter
from rqle_ai_langchain_util.llms.llm_mediator import LLMMediator

# logging configuration
from time import time
import logging
from logging.config import dictConfig
dictConfig(settings.LOG_CONFIG)
logger = logging.getLogger(__name__)

load_dotenv()


class PodProtocol:

    def info(self):
        """
        :return: the name of the class
        """
        return self.__class__.__name__

    def __init__(self, config_folder: str):
        """
        :param config_folder: the folder containing the configuration for execution
        """
        self.config_folder = config_folder
        self.parser = StrOutputParser()
        self.llm_mediator = LLMMediator(LLMAdapter.OLLAMA_AI, self.config_folder)

    def load_chain(self):
        try:
            # configure the prompt
            prompt = PromptTemplate(template=self.llm_mediator.prompt_template.prompt,
                                    input_variables=['user_input'])
            logger.debug(f'Generated prompt: {prompt}')

            # return the LLM
            chain = prompt | self.llm_mediator.model | self.parser
            return chain
        except Exception as e:
            logger.error(f'Error loading chain: {self.config_folder}\n{e}', exc_info=True)
            raise e

    def invoke_chain(self, video_file: str):
        try:
            # retrieve the text transcription of the video
            audio_filename = video_util.video_to_wav(video_file)
            transcribed_text = video_util.transcribe_audio(audio_filename, speech_recognition_engine='vosk')
            file_util.write_file(file_dir=settings.TEXT_TMP_FOLDER, file_name='transcribed_text.txt',
                                 file_content=transcribed_text)

            # invoke the chain on the transcribed text
            chain = self.load_chain()
            output = chain.invoke({'user_input': transcribed_text})
            logger.debug(f'Output from PodProtocol: {output}')
            file_util.write_file(file_dir=settings.TEXT_TMP_FOLDER, file_name='video_summary.txt',
                                 file_content=output)
            return output
        except Exception as e:
            logger.error(f'Error executing chain: {self.config_folder}\n{e}', exc_info=True)


if __name__ == '__main__':
    start = time()
    try:
        pod_protocol = PodProtocol(config_folder='pod_protocol')
        pod_protocol.invoke_chain(video_file='C:/Users/quent/Videos/Podcasts/TalkLab_interviews/EP22_AI_Episode-4.mp4')
    except Exception:
        pass
    finally:
        # calculate elapsed time
        total_time = round((time() - start) / 60, 3)
        logger.info(f'Elapsed time to generate summary: {total_time} min')
