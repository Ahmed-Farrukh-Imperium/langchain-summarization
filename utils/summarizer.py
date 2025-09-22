import os
from utils.env_loader import load_env_vars
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate

load_env_vars()

def summarizer_chain(text: str):

  deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

  llm = AzureChatOpenAI(
    azure_deployment=deployment_name,
    temperature=0
  )

  prompt_template = PromptTemplate.from_template(
    "Summarize the following text into exactly 3 sentences:\n\n{text}"
  )


  chain = prompt_template | llm 

  print("\n--- 3 Sentence Summary ---")
  print(chain.invoke({"text":text}).content)