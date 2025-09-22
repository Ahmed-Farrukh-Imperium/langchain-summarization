import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION") 

llm = AzureChatOpenAI(
    openai_api_key=api_key,
    azure_endpoint=endpoint,
    deployment_name=deployment,
    openai_api_version=api_version,
    temperature=0
)

prompt_template = PromptTemplate.from_template(
    "Summarize the following text into exactly 3 sentences:\n\n{text}"
)


chain = prompt_template | llm 

test_paragraph = """
Artificial intelligence (AI) refers to the simulation of human intelligence processes by machines, 
particularly computer systems. These processes include learning, reasoning, problem-solving, perception, 
and language understanding. AI applications today are numerous, ranging from self-driving cars to 
virtual assistants like Siri and Alexa, recommendation engines on Netflix, fraud detection systems in 
banking, and even medical diagnosis tools. One of the driving forces behind AI is machine learning, 
where systems learn and improve from experience without being explicitly programmed. Deep learning, a 
subset of machine learning, uses neural networks to model complex patterns in data and has significantly 
advanced fields such as image and speech recognition. The rapid progress in AI has sparked excitement 
as well as concern. While AI promises efficiency, cost reduction, and breakthroughs in various industries, 
it also raises ethical questions about job displacement, data privacy, decision-making bias, and the 
future of human-AI interaction. Governments, companies, and researchers are actively debating regulations 
and frameworks to ensure responsible AI development. As AI continues to evolve, society must strike a 
balance between harnessing its potential and addressing its challenges. This dual nature of AI makes it 
one of the most transformative and debated technologies of our time.
"""

print("\n--- 3 Sentence Summary ---")
print(chain.invoke({"text":test_paragraph}).content)

prompt_template_one = PromptTemplate.from_template(
    "Summarize the following text into exactly 1 sentence:\n\n{text}"
)

chain_one = prompt_template_one | llm

print("\n--- 1 Sentence Summary ---")
print(chain_one.invoke({"text":test_paragraph}).content)