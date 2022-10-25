import os
os.system("pip install gradio==3.3")
import gradio as gr
import numpy as np
import streamlit as st

title = "SpeechMatrix Speech-to-speech Translation"

description = "Gradio Demo for SpeechMatrix. To use it, simply record your audio, or click the example to load. Read more at the links below. \nNote: These models are trained on SpeechMatrix data only, and meant to serve as a baseline for future research."

article = "<p style='text-align: center'><a href='https://research.facebook.com/publications/speechmatrix' target='_blank'>SpeechMatrix</a> | <a href='https://github.com/facebookresearch/fairseq/tree/ust' target='_blank'>Github Repo</a></p>"

SRC_LIST = ['cs', 'de', 'en', 'es', 'et', 'fi', 'fr', 'hr', 'hu', 'it', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl']
TGT_LIST = ['en', 'fr', 'es']
MODEL_LIST = ['xm_transformer_sm_all-en']
for src in SRC_LIST:
    for tgt in TGT_LIST:
        if src != tgt:
            MODEL_LIST.append(f"textless_sm_{src}_{tgt}")
        
examples = []

io_dict = {model: gr.Interface.load(f"huggingface/facebook/{model}", api_key=st.secrets["api_key"]) for model in MODEL_LIST}
   
def inference(audio, model):
    out_audio = io_dict[model](audio)   
    return out_audio 
gr.Interface(
    inference,
    [gr.inputs.Audio(source="microphone", type="filepath", label="Input"),gr.inputs.Dropdown(choices=MODEL_LIST, default="xm_transformer_sm_all-en",type="value", label="Model")
],
    gr.outputs.Audio(label="Output"),
    article=article,
    title=title,
    examples=examples,
    cache_examples=False,
    description=description).queue().launch()