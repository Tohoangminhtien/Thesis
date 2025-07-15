from agno.agent import Agent
from agno.models.google.gemini import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
import re
import json
import pandas as pd

key = "AIzaSyBLLXxkA0Ij5iHr6tXgIcpwzCBANJQn4_o"
model = Gemini(id="gemini-2.0-flash-exp", api_key=key)
agent = Agent(model=model, markdown=True)


analysis_prompt = """"""

aggr_prompt = """Bạn là một tác nhân điều phối, chịu trách nhiệm tổng hợp thông tin từ các tác nhân khác để phân loại một tuyên bố thành một trong sáu nhãn: pants-fire, false, barely true, half-true, mostly true, true. Bạn nhận đầu ra từ ba tác nhân:
- Tác nhân phân loại văn bản: {"label": "nhãn dự đoán", "confidence": 0.0-1.0}
- Tác nhân phân tích siêu dữ liệu: {"credibility": 0.0-1.0}
- Tác nhân xác định độ chính xác: {"accuracy": 0.0-1.0}

Dựa trên các đầu vào này, áp dụng logic sau để chọn nhãn cuối cùng:
1. Bắt đầu với nhãn từ Tác nhân phân loại văn bản.
2. Nếu điểm độ tin cậy từ Tác nhân phân tích siêu dữ liệu < 0.5, điều chỉnh nhãn về phía "false" (ví dụ, từ half-true thành barely true).
3. Nếu điểm accuracy từ Tác nhân xác định độ chính xác > 0.7, củng cố nhãn về phía "true"; nếu < 0.3, củng cố về phía "false"
4. Nếu có mâu thuẫn giữa các tác nhân, ưu tiên điểm accuracy nếu confidence < 0.7

Trả về kết quả dưới dạng JSON: {"final_label": "nhãn cuối cùng"}. Ví dụ: {"final_label": "barely true"}.

Đầu vào:
- Tuyên bố: {statement}
- Siêu dữ liệu: {metadata}
- Đầu ra từ các tác nhân: 
    Classification Agent: {text_classifier_output}
    Analysis Metadata Agent: {metadata_analyzer_output}
    Fact Checker Agent {accuracy_evaluator_output}"""

evaluate_prompt = """Bạn là một chuyên gia xác định độ chính xác. Nhiệm vụ của bạn là phân tích một tuyên bố và so sánh nó với các kết quả tìm kiếm web để xác định mức độ chính xác. Các kết quả tìm kiếm bao gồm tiêu đề và nội dung tóm tắt. Hãy thực hiện các bước sau:
1. Đọc kỹ tuyên bố và các kết quả tìm kiếm.
2. Đánh giá xem tuyên bố có được hỗ trợ, mâu thuẫn, hay không rõ ràng dựa trên nội dung tìm kiếm.
3. Gán điểm số độ chính xác từ 0.0 (hoàn toàn sai) đến 1.0 (hoàn toàn đúng) dựa trên mức độ phù hợp và tính xác thực.
4. Nếu thông tin không đủ hoặc mâu thuẫn, đưa ra điểm số hợp lý dựa trên suy luận.

Trả về kết quả dưới dạng JSON: {{'accuracy': số thực từ 0.0 đến 1.0}}. Ví dụ: {{'accuracy': 0.8}}.

Tuyên bố: {statement}
Kết quả tìm kiếm: {search_results}"""


def ClassifyAgent(agent, prompt, statement: str):
    # OK
    result = agent.run(message=prompt + "\n" + statement)
    return json.loads(result.content)


def AnalysisAgent(agent, prompt, metadata: dict):
    # Ok
    text = ""
    for k, v in metadata.items():
        text += f"{k}: {v}\n"
    result = agent.run(message=prompt + "\n" + text)
    match = re.search(r'\{.*\}', result.content)
    return json.loads(match.group())


def SearchAgent(statement: str) -> str:
    tool = DuckDuckGoTools()
    result = tool.duckduckgo_search(statement, max_results=3)
    result = json.loads(result)
    temp = []
    for i in result:
        temp.append({"title": i['title'], "content": i["body"]})
    return temp


def FactCheckerAgent(agent, prompt: str, statement: str, search_results: list):
    result = agent.run(message=prompt.format(
        statement=statement, search_results=search_results))
    match = re.search(r'\{.*\}', result.content)
    return json.loads(match.group())


def AggrAgent(agent, prompt: str, statement: str, metadata: dict, text_classifier_output: dict, metadata_analyzer_output: dict, accuracy_evaluator_output: dict):
    result = agent.run(message=prompt.format(statement=statement, metadata=metadata, text_classifier_output=text_classifier_output,
                       metadata_analyzer_output=metadata_analyzer_output, accuracy_evaluator_output=accuracy_evaluator_output))
    return result.content


df = pd.read_csv("data/test.csv")
# print(df.columns)


for i in range(len(df)):
    statement = df['statement'][i]
    subject = df['subject'][i]
    job_title = df['job_title'][i]
    speaker = df['speaker'][i]
    party_affiliation = df['party_affiliation'][i]
    state_info = df['state_info'][i]
    context = df['context'][i]
    label = df['label'][i]
    metadata_raw = {"Subject": subject,
                    "Speaker": speaker,
                    "Job title": job_title,
                    "State info": state_info,
                    "Party_affiliation": party_affiliation,
                    "Context": context
                    }
    metadata = {k: v for k, v in metadata_raw.items() if pd.notna(v)
                and v != ''}
    # --------------------
    
    
