# here we preporcess the audio file to preprocess audio file

# import librosa
from pydub import AudioSegment

# Create an AudioSegment instance
wav_file = AudioSegment.from_file(file="voice_data",format="mp3")

# Check the type
print(type(wav_file))
print(wav_file.channels)
