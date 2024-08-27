import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from langchain.embeddings.huggingface import  HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import  LangchainEmbedding
from dataclasses import dataclass
import os, sys
from src.log import logging
# from ids.secret import  api_key
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

            documents = SimpleDirectoryReader(input_dir="RAG_data")

            logging.info("Document loading...")
            documents = documents.load_data()
            logging.info("Document Successfully loaded")

            system_prompt = """
                You are a Question&Answer assistant, your goal is to answer questions based on instructions and context provided.
                                """

            prompt_format = SimpleInputPrompt("<|USER|>{user_query}<|ASSISTANCE|>") #format for LLamaIndex



            # uploading the model Here we can use any model, each model has its own parameters
            llm = HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=256,
                generate_kwargs={"temperature": 0.0, "do_sample": False},
                system_prompt=system_prompt,
                query_wrapper_prompt=prompt_format,
                tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
                model_name="meta-llama/Llama-2-7b-chat-hf",
                device_map="auto",
            )

            """
             Embeddding is important as it transforms the textual data in vectors efficiently 
             we import embedding for langchain and llama-2 
             Service Context to wrap both the llm making it more efficient
             
            """


            # successfully added embedding to model
            embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
            # this  model efficiently maps sentences to para --> dense vector space
            # These Sparse vectors are populated with information and can be efficiently stored
            service_context = ServiceContext.from_defaults(
                chunk_size=1024,
                llm=llm,
                embed_model=embed_model
            )

            logging.info("llms")
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)

            query_engine = index.as_query_engine()

            logging.info("Preparing the query output")

            response = query_engine.query("hey man how are you doing?")

            logging.info("Result generated successfully")

            return response

        except Exception as e:
            raise CustomException(e,sys)

        finally:
            pass


if __name__ == "__main__":
    object = RAG()
    return_value = object.get_query_output()
    print(return_value)