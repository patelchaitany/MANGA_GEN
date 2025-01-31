import textgrad as tg
import os
import pandas as pd
import numpy as np
import json

import langchain
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, RemoveMessage,AIMessage,HumanMessage,BaseMessage,FunctionMessage
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq

import time
from prompt import writer_prompt

from langchain.output_parsers import PydanticOutputParser
from typing import List
from pydantic import BaseModel, Field
import tqdm
import pprint

os.environ['GOOGLE_API_KEY'] = 'AIzaSyCE7RJhTM6Il1Fbf7zr_jsIhfSOLKTga14'
os.environ["GROQ_API_KEY"] = "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"

#model = ChatGroq(temperature=0, groq_api_key="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", model_name="llama-3.3-70b-versatile")

class responce(BaseModel):
    content:List[str] = Field(...,description="List Of Generated Story Ideas")


def generate_Example():
    prompt = ChatPromptTemplate.from_messages([
        ("system",writer_prompt),
        MessagesPlaceholder(variable_name="messages")]
    )
    
    chain =prompt | model 
    return chain

def get_topic():
    
    parser = PydanticOutputParser(pydantic_object=responce)
    format_instructions = parser.get_format_instructions()
    prompt_m = ChatPromptTemplate.from_messages([
        ("system","Please provide story idea for writing story given user requested topic. \nGive Responce in JSON with this structure everthing follow the given structure and do give anything other than json.There should be no output before json and after json. \n"),
        ("system","{format_instructions}")
        ,MessagesPlaceholder(variable_name="messages")
    ])
    prompt_m = prompt_m.partial(format_instructions=format_instructions)

    chain = prompt_m |model| parser
    return chain


def prompt_optimization(data):

    llm_engine = tg.get_engine("gemini-2.0-flash-exp")
    tg.set_backward_engine(llm_engine)
    
    STARTING_SYSTEM_PROMPT = writer_prompt
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True, 
                            role_description="structured system prompt to a somewhat capable language model that will write a story given a topic.")
    model = tg.BlackboxLLM(llm_engine, system_prompt)
    optimizer = tg.TextualGradientDescent(engine=llm_engine, parameters=[system_prompt])
    evaluation_instruction = (
                           "Evaluate any given answer to this question, "
                           "be smart, logical, and very critical. "
                           "Just provide concise feedback.")

    loss_system_prompt = "You are a smart language model that evaluates story from given topic. You do not propose story, only evaluate existing story critically and give very concise feedback on system_prompt."
    loss_system_prompt = tg.Variable(loss_system_prompt, requires_grad=False, role_description="system prompt to the loss function")
    instruction = """
    Think About the story topic."""

    format_string = "{instruction}\nStory Topic: {{topic}}\nCurrent story: {{responce_story}}.\n"
    format_string = format_string.format(instruction=instruction)
    fields = {"topic":None, "responce_story":None}
    formatted_llm_call = tg.autograd.FormattedLLMCall(engine=llm_engine,
                                                  format_string=format_string,
                                                  fields=fields,
                                                  system_prompt=loss_system_prompt)
    def loss_fn(question:tg.Variable, answer:tg.Variable,responce:tg.Variable)->tg.Variable:
        input = {"topic":question,"responce_story":responce}

        return formatted_llm_call(inputs = input,response_role_description=f"evaluation of the {system_prompt.get_role_description()}")

    with open("prompt.jsonl","w") as f:
        for epoch in range(3):
            loss_f = []
            optimizer.zero_grad()
            for i in tqdm.tqdm(data):
                question_t = i["topic"]
                answer_t = i["story"]
                question = tg.Variable(question_t,requires_grad=False,role_description="topic")
                answer = tg.Variable(answer_t,requires_grad=False,role_description="actual story")
                result = model(question)
                result.requires_grad = False
                loss = loss_fn(question = question,answer=answer,responce=result)
                loss_f.append(loss)
            total_loss = tg.sum(loss_f)
            total_loss.backward()
            optimizer.step()
            print((system_prompt.value))
            f.write(json.dumps({"epoch":epoch,"system_prompt":system_prompt.value}))
            f.write("\n")

def main():

    story = generate_Example()
    idea = get_topic()
    with open("topic.jsonl","x") as f:

        while True:
            topic = input("Enter Topic: ")
            print(f"Topic is {topic}")
            msg = [HumanMessage(content=f"Topic is {topic}")]
            responce = idea.invoke(msg)
            Topic = responce.content
            for i in Topic:
                msg1 = [HumanMessage(content=f"write story for this topic:{i}")]
                gen_story = story.invoke(msg1)
                time.sleep(2)
                f.write(json.dumps({"topic":i,"story":gen_story.content}))
                f.write("\n")
        print(responce)

if __name__ == "__main__":

    data = []

    with open("prompt2.jsonl") as f:
        
        data_temp = f.readlines()
        for i in data_temp:
            data.append(json.loads(i))

    pprint.pprint(data[0]["system_prompt"])
    
