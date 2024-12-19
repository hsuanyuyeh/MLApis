import torch
import os
import gradio as gr
from transformers import pipeline
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


llm = OllamaLLM(model="llama3")
temp = '''
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
'''

pt = PromptTemplate(
    input_variables=["context"],
    template = temp
)

prompt_to_LLAMA3 = LLMChain(llm=llm, prompt=pt)

def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30
    )
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    result = prompt_to_LLAMA3(transcript_txt)
    return result['text'] 

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(
    fn=transcript_audio, 
    inputs=audio_input, outputs=output_text,
    title="Audio Transcription App",
    description="Upload the audio file"
)
iface.launch(server_name="0.0.0.0", server_port=7860)