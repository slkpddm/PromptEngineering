import os
from langchain import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from constants import openai_key
import streamlit as st

os.environ['OPENAI_API_KEY']=openai_key

st.title("Language Translation")
input1 = st.text_input("Enter Setence to translate")
input2 = st.text_input("Enter the language to translate")


template='''In an easy way translate the following sentence '{sentence}' into {target_language}'''
language_prompt = PromptTemplate(
    input_variables=["sentence",'target_language'],
    template=template
)
language_prompt.format(sentence="How are you",target_language='hindi')

llm=OpenAI(temperature=0.7)
chain=LLMChain(llm=llm,prompt=language_prompt)

if input1 and input2:
    st.write(chain({'sentence':input1,'target_language':input2}))