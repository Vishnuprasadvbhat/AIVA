import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from dataclasses import dataclass
import librosa
import os, sys
from src.log import logging
from src.exception import CustomException

@dataclass
class VoiceInputConfig:
    output_file_path = str=os.path.join('query_data') #add this later

class VoicetoText():
    def __init__(self):
        self.file_path = VoiceInputConfig()
    def convert_text(self,input_file):
        logging.info('Audio file entered conversion method')

        try:

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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
            raise CustomException(e,sys)


if __name__ == "__main__":

    object = VoicetoText()
    input_file = "voice_data/there_was_a_time-32907.mp3"
    object.convert_text(input_file)
