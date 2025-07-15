from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
import json


# def search_google(query: str) -> None:
#     """
#     Use this function to get infomation from Google Search

#     Args:
#         query (str): xxx

#     Returns:
#         str: xxx
#     """
#     tool = GoogleSearchTools()
#     result = tool.google_search(query)
#     result = json.loads(result)
#     for i in result:
#         print("------------------------------")
#         print(i['title'])
#         print(i['description'])

#     print("------------------------------")


def search_duckduckgo(query: str) -> str:
    """
    Use this function to look up infomation.

    Args:
        query (str): content

    Returns:
        str: 5 similarity result
    """
    tool = DuckDuckGoTools()
    result = tool.duckduckgo_search(query)
    result = json.loads(result)
    temp = []
    for i in result:
        temp.append({"title": i['title'], "content": i["body"]})
    return temp


result = search_duckduckgo(
    "Ngày 6.5, Công an thị xã Phước Long, tỉnh Bình Phước đã tạm giữ bà Lê Thị Xuyến (46 tuổi, ngụ phường Long Phước, thị xã Phước Long) để điều tra về việc bắt cóc trẻ em.")
print(result)
