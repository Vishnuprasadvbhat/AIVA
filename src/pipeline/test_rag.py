import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, disk_offload
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from dataclasses import dataclass
import os, sys
from src.log import logging
from src.exception import CustomException

@dataclass
class RagConfig:
    rag_ouput = str = os.path.join("folder_name", "file_name")


class RAG:
    def __init__(self):
        self.output_path = RagConfig()

    def get_query_output(self):

        try:
            logging.info("Entered RAG playground")

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            with open("query_data/output.txt", 'r') as file:
                user_query = file.readlines()

            documents = SimpleDirectoryReader(input_dir="RAG_data")

            logging.info("Document loading...")
            documents = documents.load_data()
            logging.info("Document Successfully loaded")

            system_prompt = """
                You are a Question&Answer assistant, your goal is to answer questions based on instructions and context provided.
            """

            prompt_format = SimpleInputPrompt("<|USER|>{user_query}<|ASSISTANCE|>")  # format for LLamaIndex

            repo_id = "meta-llama/Llama-2-7b-chat-hf"

            # Initialize an empty model
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )

            # Load model and apply disk offloading
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=repo_id,
                device_map='auto',
                offload_folder=r"C:\Users\vishn\AIVA"
            )

            # Apply disk offloading
            model = disk_offload(model, offload_dir=r"C:\Users\vishn\AIVA")

            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                use_fast=True
            )

            # Create pipeline
            pipe = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                max_length=512
            )

            llm = HuggingFaceLLM(
                pipeline=pipe,
                context_window=4096,
                max_new_tokens=256,
                generate_kwargs={"temperature": 0.0, "do_sample": False},
                system_prompt=system_prompt,
                query_wrapper_prompt=prompt_format,
                device_map="auto"
            )

            embedded_model = Settings.embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

            Settings.chunk_size = 1024,
            Settings.llm = HuggingFaceLLM(model_name="meta-llama/Llama-2-7b-chat-hf")

            logging.info("llms")
            index = VectorStoreIndex.from_documents(documents, embedded_model=embedded_model)

            query_engine = index.as_query_engine()

            logging.info("Preparing the query output")

            response = query_engine.query("hey man how are you doing?")

            logging.info("Result generated successfully")

            return response

        except Exception as e:
            raise CustomException(e, sys)

        finally:
            pass


if __name__ == "__main__":
    object = RAG()
    return_value = object.get_query_output()
    print(return_value)
