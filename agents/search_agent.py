from agno.tools.googlesearch import GoogleSearchTools
import json
import warnings

warnings.filterwarnings("ignore")


class SearchAgent:
    def __init__(self, max_results: input = 3):
        self.tool = GoogleSearchTools()
        self.max_results = max_results

    def search(self, query: str) -> None:
        response = self.tool.google_search(query, self.max_results)
        response = json.loads(response)
        result = ""
        for i in response:
            if i["title"]:
                result += f"Title: {i['title']} \nDescription: {i['description']}\n"
        return result
