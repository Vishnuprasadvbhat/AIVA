# import librosa
from pydub import AudioSegment

# Create an AudioSegment instance
wav_file = AudioSegment.from_file(file="Best FREE Speech To Text is NOW FASTER!!!.mp3",format="mp3")

# Check the type
print(type(wav_file))