device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

with open("query_data/output.txt", 'r') as file:
    user_query = file.readlines()

repo_id = "meta-llama/Llama-2-7b-chat-hf"

documents = SimpleDirectoryReader(input_dir="RAG_data")

logging.info("Document loading...")
documents = documents.load_data()
logging.info("Document Successfully loaded")

system_prompt = """
                You are a Question&Answer assistant, your goal is to answer questions based on instructions and context provided.
                                """

prompt_format = SimpleInputPrompt("<|USER|>{user_query}<|ASSISTANCE|>")  # format for LLamaIndex

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True).cpu()
disk_offload(model=model, offload_dir="offload")

pipeline = transformers.pipeline(
    "text-generation",
    model=repo_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map='auto',
    offload_state_dict=True
)
llm = HuggingFaceLLM(
    pipeline=pipeline,
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=prompt_format,
    # tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    # model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",

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
