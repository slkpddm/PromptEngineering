import os
from langchain import OpenAI
from constants import openai_key 
from langchain.chains import LLMChain
from langchain import PromptTemplate
import streamlit as st

os.environ['OPENAI_API_KEY']=openai_key

st.title("Prompt Engineering using Langchain")
input_text = st.text_input("Enter a Topic")

# Prompt Templates
demo_template='''I want you to act as a acting financial advisor for people.
In an easy way, explain the basics of {financial_concept}.'''
prompt=PromptTemplate(
    input_variables=['financial_concept'],
    template=demo_template
)

llm=OpenAI(temperature=0.7)
chain=LLMChain(llm=llm,prompt=prompt)

if input_text:
    st.write(chain.run(input_text))


