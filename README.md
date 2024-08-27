# This project involves designing an End-to-End AI Voice Assistance Pipeline

## Objective:
<h3>
Design a pipeline that takes a voice query command, converts it into text, uses a Large Language Model (LLM) to generate a response, and then converts the output text back into speech. The system should have low latency, Voice Activity Detection (VAD), restrict the output to 2 sentences, and allow for tunable parameters such as pitch, male/female voice, and speed.
</h3>

## Architecture: 

![AIVA drawio](https://github.com/user-attachments/assets/a880910f-1c97-409f-be14-9f1ff1cb58a5)


<H2> Step 1: Voice-to-Text Conversion </H2>
<p>
    Using https://huggingface.co/distil-whisper/distil-medium.en we convert the Audio file and save it under root directory.
</p>
