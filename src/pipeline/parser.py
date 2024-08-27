from src.log import  logging
from src.exception import CustomException
import sys,os
from dataclasses import dataclass
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from src.utils import open_file
from IPython.display import Audio

@dataclass
class TTS:
    def __init__(self):
        self.output_path = str=os.path.join()

    def convert_to_speech(self, input_filepath:str, tone:str):
        try:

            input_file = open_file(input_filepath)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
            tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

            tone = str(tone.lower())
            if tone == "female":
                description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
            elif tone == "male":
                description = "A male speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

            else:
                print(f'Invalid Input {tone}')


            input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tokenizer(input_file, return_tensors="pt").input_ids.to(device)

            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            audio_arr = generation.cpu().numpy().squeeze()
            audio_data = sf.write("speech.wav", audio_arr, model.config.sampling_rate)

            return audio_data

        except Exception as e:
            raise CustomException(e,sys)



if __name__ == "__main__":
    pass

