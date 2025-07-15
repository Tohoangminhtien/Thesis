from agno.tools.wikipedia import WikipediaTools
from agno.agent import Agent
from agno.models.google.gemini import Gemini
import json
import pandas as pd


def authenticate_llm(name: str, job: str) -> str:
    """
    Use this function to authenticate name and job.

    Args:
        job (str): job
        name (str): person name 

    Returns:
        str: result
    """
    key = "AIzaSyBLLXxkA0Ij5iHr6tXgIcpwzCBANJQn4_o"
    model = Gemini(id="gemini-2.0-flash-exp", api_key=key)
    agent = Agent(model=model, markdown=True)
    result = agent.print_response(
        f"Does {name} have a position as {job}, answer yes/no/i don't know", stream=True)
    return result


def authenticate_job(name: str, job: str) -> str:
    """
    Use this function to authenticate name and job.

    Args:
        job (str): job
        name (str): person name 

    Returns:
        str: result
    """
    tool = WikipediaTools()
    result = tool.search_wikipedia(job + " " + name)
    result = json.loads(result)['content']
    result = result.split(".")[:4]
    return ".".join(result)


df = pd.read_csv('data/test.csv')
names = df['speaker']
jobs = df['job_title']
for name, job in zip(names, jobs):
    if name and job:
        authenticate_llm(name, job)
    else:
        print(f"Name: {name} - Job {job}")
