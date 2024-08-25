import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer
from llama_index.llms.huggingface import  HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from dataclasses import dataclass
import os, sys
from src.log import logging
from ids.secret import  api_key
from src.exception import CustomException


@dataclass
class RagConfig:
    rag_ouput = str=os.path.join("folder_name", "file_name")


class RAG:
    def __init__(self):
        self.output_path = RagConfig()

    def get_query_output(self):

        try:
            logging.info("Entered RAG playground")

            with open("query_data/output.txt",'r') as file:
                user_query = file.readlines()

            documents = SimpleDirectoryReader("RAG_data/Human 1.pdf").load_data()

            system_prompt = """
                You are a Question&Answer assistant, your goal is to answer questions based on instructions and context provided.
                                """

            prompt_format = SimpleInputPrompt("<|USER|>{user_query}<|ASISSTANCE|>")

        except Exception as e:
            raise CustomException(e,sys)






if __name__ == "__main__":
    object = ""
