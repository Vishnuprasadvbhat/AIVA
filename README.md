# This project involves designing an End-to-End AI Voice Assistance Pipeline

## Objective:
<h3>
Design a pipeline that takes a voice query command, converts it into text, uses a Large Language Model (LLM) to generate a response, and then converts the output text back into speech. The system should have low latency, Voice Activity Detection (VAD), restrict the output to 2 sentences, and allow for tunable parameters such as pitch, male/female voice, and speed.
</h3>

## Architecture: 

![AIVA drawio](https://github.com/user-attachments/assets/a880910f-1c97-409f-be14-9f1ff1cb58a5)

<h2>Approach </h2>

<p> The goal is to implement VOICE-TO-VOICE Assistance Pipeline using various open source models, combing their power building a voice Assistance Tool. <br> </p>


<H2> Step 1: Voice-to-Text Conversion </H2>
<h4> Here we are using ` distil-whisper/distil-medium.en ` for conversion of speech to text Efficiently.</h4>

[distll-whisper](https://huggingface.co/distil-whisper/distil-medium.e)

<h4>Why only distill-whisper? </h4>
<ol> 
<li> Able to Convert the text faster and with less memory compared to Wishper AI and  its quantized models </li>
<li> Comsumes less computation power</li>
<li> State of Art WRE of 8 for ` medium-en ` model</li>
</ol>

## Here is the code walkaround of Step-1 

#### Defining a function to convert the voice-to-text data

`     def convert_text(self,input_file):
        logging.info('Audio file entered conversion method')         
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32 `

            model_id = "distil-whisper/distil-medium.en"

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )

            model.to(device)

            processor = AutoProcessor.from_pretrained(model_id)
            logging.info("Model is set")
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=15,
                batch_size=16,
                torch_dtype=torch_dtype,
                device=device,
            )


            logging.info('Converison Initiated...')


            result = pipe(input_file)
            text_data = result['text']
            logging.info('Voice successfully Converted to text')


            folder_path = self.file_path.output_file_path

            # Create the folder if it doesn't exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Save the file in the created folder
            with open(os.path.join(folder_path, "output.txt"), "w") as file:
                file.write(text_data)

            logging.info('Text file saved successfully')

            return text_data

        except Exception as e:
            raise CustomException(e,sys) `
            
#### To save the file in a specific folder we have defined a class variable: 

` @dataclass
class VoiceInputConfig:
    output_file_path = str=os.path.join('query_data') #add this later
class VoicetoText():
    def __init__(self):
        self.file_path = VoiceInputConfig() 
` 
        
<H2> Step 2: Building RAG using llama-2 and llamaIndex </H2>

[meta-llama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

 <h3> Why RAG? </h3>
 <strong> <p> We could have directly used any text generation llms such as langchain or mistral-7b but llms results in irrelevant response that may be out of context. <br> 
 Due to this our responses may to not be up to the mark. By implementing RAG using LLamaIndex (optimal for retrival and indexing operations)  We make sure thwe outputs are accurate to the query <br>
    
RAG document: [human_01](https://www.kaggle.com/datasets/projjal1/human-conversation-training-data)

<h6>Required Imports</h6>

` import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from langchain.embeddings.huggingface import  HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import  LangchainEmbedding `


<h6> For LlamaIndex we have specific format to query the data</h6>

`  prompt_format = SimpleInputPrompt("<|USER|>{user_query}<|ASSISTANCE|>") #format for LLamaIndex `

##### Now we call the Model using implementing LLM Pipeline, here we define all the required parameters and model name etc 

` llm = HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=256,
                generate_kwargs={"temperature": 0.0, "do_sample": False},
                system_prompt=system_prompt,
                query_wrapper_prompt=prompt_format,
                tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
                model_name="meta-llama/Llama-2-7b-chat-hf",
                device_map="auto",
            ) 
`
We have included embedding layer to the 
</p></strong>
<ol>
    <li>  </li>
    
</ol>

