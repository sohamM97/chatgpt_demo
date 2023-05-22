import dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

dotenv.load_dotenv("../.env")

llm = OpenAI(temperature=0)

tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

prompt = input("The Wikipedia research task: ")

agent.run(prompt)
