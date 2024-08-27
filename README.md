# This project involves designing an End-to-End AI Voice Assistance Pipeline

## Objective:
<h3>
Design a pipeline that takes a voice query command, converts it into text, uses a Large Language Model (LLM) to generate a response, and then converts the output text back into speech. The system should have low latency, Voice Activity Detection (VAD), restrict the output to 2 sentences, and allow for tunable parameters such as pitch, male/female voice, and speed.
</h3>

## Architecture: 

![AIVA drawio](https://github.com/user-attachments/assets/a880910f-1c97-409f-be14-9f1ff1cb58a5)

<h1>Approach </h1>

<p> The goal is to implement VOICE-TO-VOICE Assistance Pipeline using various open source models, combing their power building a voice Assistance Tool. <br>

<ol> 
    <h3> Here we are using `distil-whisper/distil-medium.en` for conversion of speech to text Efficiently.</h3>
    <h4>Wht distill-whisper? </h4>
<li> Able to Convert the text faster compared to Wishper AI model </li>
<li> Comsumes less computation power</li>
<li> State of Art WRE of 8 for a ```medium-en model```</li>
 
    
</ol>`


</p>
<H2> Step 1: Voice-to-Text Conversion </H2>
<p>
    Using https://huggingface.co/distil-whisper/distil-medium.en we convert the Audio file and save it under root directory.
</p>
