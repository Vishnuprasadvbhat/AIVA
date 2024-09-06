import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer ,AutoModelForCausalLM, AutoConfig
from accelerate import disk_offload
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, Settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from dataclasses import dataclass
import os,sys
from src.log import logging
# from ids.secret import  api_key
from src.exception import CustomException
import transformers


@dataclass
class RagConfig:
    rag_ouput = str=os.path.join("rag_output", "rag_res.txt")


class RAG:
    def __init__(self):
        self.output_path = RagConfig()

    def get_query_output(self):

        try:
            logging.info("Entered RAG playground")

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


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

            repo_id = "meta-llama/Llama-2-7b-chat-hf"
            config = AutoConfig.from_pretrained(repo_id)

            with torch.inference_mode():
                model = AutoModelForCausalLM.from_config(config)

            device_map = "auto"
            needs_offloading = torch.cuda.is_available()

            logging.info("Disk off-loaded")


            if needs_offloading:
                # Load model with offloading enabled
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    device_map=device_map,
                    offload_folder="offload",
                    offload_state_dict=True,
                    torch_dtype=torch_dtype
                )
            else:
                # Load model to CPU or GPU without offloading
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    torch_dtype=torch_dtype
                ).to(device)

            logging.info("Creating pipeline")

            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map=device_map
            )
            logging.info("Pipeline Created")

            """
            Why disk_offload?..
            
            This error arises when there’s no enough memory to load the model. To address this, 
            the error message suggests using the disk_offload function instead of directly loading the entire model to memory. 
            The disk_offload function is specifically designed to handle the offloading of model components to disk in a 
            more memory-efficient manner."""


            # uploading the model Here we can use any model, each model has its own parameters

            llm = HuggingFaceLLM(
                pipeline=pipeline,
                context_window=4096,
                max_new_tokens=256,
                generate_kwargs={"temperature": 0.0, "do_sample": False},
                system_prompt=system_prompt,
                query_wrapper_prompt=prompt_format,
            )
            logging.info("llm Initialized")

            """
             Embeddding is important as it transforms the textual data in vectors efficiently 
             we import embedding for langchain and llama-2 
             Service Context to wrap both the llm making it more efficient
             
            """

            # This has been depriciated
                # Before:
            """embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))"""

                #After

            logging.info("Embedding the model")


            embedded_model = Settings.embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

            logging.info("Embedded the model")

            """ 
                This  model efficiently maps sentences to para --> dense vector space.
             These Sparse vectors are populated with information and can be efficiently stored
            """

            Settings.chunk_size = 1024
            Settings.llm = llm


            logging.info("Settings comfirmed")

            index = VectorStoreIndex.from_documents(documents, embedded_model = embedded_model)


            logging.info("Preparing the query output")

            query_engine = index.as_query_engine()

            logging.info("LOADING RESPONSE")

            response = query_engine.query("hey man how are you doing?")

            logging.info("Result generated successfully")

            return response[0]

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    object = RAG()
    return_value = object.get_query_output()
    print(return_value)