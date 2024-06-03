import gzip
import json
import logging
import os
import queue
import re
import threading
import time

import flask
from genai import Client, Credentials
from genai.schema import TextGenerationParameters
import jwt
from langchain.prompts import PromptTemplate
import requests

WD_API_URL = os.getenv("WD_API_URL")
WD_API_KEY = os.getenv("WD_API_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
GENAI_API = os.getenv("GENAI_API")
GENAI_KEY = os.getenv("GENAI_KEY")

# Enrichment task queue
q = queue.Queue()

app = flask.Flask(__name__)
app.logger.setLevel(logging.INFO)
app.logger.handlers[0].setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s (%(filename)s:%(lineno)d)"
    )
)

credentials = Credentials(api_endpoint=GENAI_API, api_key=GENAI_KEY)
client = Client(credentials=credentials)
model_id = "ibm/granite-8b-japanese"
parameters = TextGenerationParameters(
    max_new_tokens=512,
    min_new_tokens=1,
    repetition_penalty=1.05,
    stop_sequences=[
        "<|endoftext|>",
    ],
    decoding_method="greedy",
)

system_prompt_template = """### System:
あなたは誠実で優秀なAIアシスタントです。ユーザーの指示に可能な限り正確に従ってください。
### User:
{prompt}
### Assistant:
"""
prompt_template = PromptTemplate(
    input_variables=[
        "prompt",
    ],
    template=system_prompt_template,
)


def create_chunks(bodies: list[str]):
    pattern = r"(?<=[\u3000-\u9FFF])\s(?=[\u3000-\u9FFF])"
    normalized_bodies = [re.sub(pattern, "", body) for body in bodies]
    chunks = []
    prev = ""
    for body in normalized_bodies:
        if body[-1] in ("。", "！", "？", "．"):
            chunks.append(prev + body)
            prev = ""
        else:
            prev += body
    else:
        if prev:
            chunks.append(prev)
    return chunks


def summarize(chunks: list[str]):
    prompts = []
    for chunk in chunks:
        prompt = prompt_template.format(
            prompt=f"以下の文章を、短く簡潔に要約してください。\n{chunk}"
        )
        prompts.append(prompt)
    summaries = [
        res.results[0].generated_text
        for res in client.text.generation.create(
            model_id=model_id, inputs=prompts, parameters=parameters
        )
    ]
    summary = "\n".join([f"- {s}" for s in summaries])
    prompt = prompt_template.format(
        prompt=f"以下の箇条書きは、ニュース記事の要約です。このニュース記事に簡潔で短い見出しタイトルをつけてください。\n{summary}"
    )
    response = client.text.generation.create(
        model_id=model_id, inputs=prompt, parameters=parameters
    )
    title = next(response).results[0].generated_text
    return summary, title


entity_system_prompt_template = """### System:
あなたは誠実で優秀なAIアシスタントです。ユーザーの指示に可能な限り正確に従ってください。
### User:
以下の文章から、IBMと協業するパートナー企業の名前を抽出してください。存在しない場合は「不明」と出力してください。
IBMは、年次イベント「THINK」において、Amazon Web Services（AWS）社と協力し、AIとデータのプラットフォームであるwatsonx内のIBM製品の全ポートフォリオをAWSサービスで利用可能になることを発表しました。これにより、エンドツーエンドのガバナンスに支えられた、オープンでハイブリッドなアプローチを通じ、より容易に企業におけるAIの活用を拡張できるようになります。
### Assistant:
企業の名前:
Amazon Web Services（AWS）
### User:
以下の文章から、IBMと協業するパートナー企業の名前を抽出してください。存在しない場合は「不明」と出力してください。
当報道資料は、2024年5月21日（現地時間）にIBM Corporationが発表したプレスリリースの抄訳をもとにしています。原文はこちらを参照ください。IBM、ibm.com、watsonx、watsonx.governance、watson.ai、watson.dataは、 米国やその他の国におけるInternational Business Machines Corporationの商標または登録商標です。他の製品名およびサービス名等は、それぞれIBMまたは各社の商標である場合があります。現時点での IBM の商標リストについては、ibm.com/trademarkをご覧ください。
### Assistant:
企業の名前:
不明
### User:
以下の文章から、IBMと協業するパートナー企業の名前を抽出してください。存在しない場合は「不明」と出力してください。
{context}
### Assistant:
企業の名前:
"""
entity_prompt_template = PromptTemplate(
    input_variables=[
        "context",
    ],
    template=entity_system_prompt_template,
)


def extract_entities(contexts: list[str]):
    entity_values = set()
    for context in contexts:
        prompt = entity_prompt_template.format(context=context)
        response = client.text.generation.create(
            model_id=model_id, inputs=prompt, parameters=parameters
        )
        entity = next(response).results[0].generated_text
        if ("不明" not in entity) and (not re.match(r"IBM.*", entity)):
            entity_values.add(entity)
    return [{"text": e, "type": "パートナー企業"} for e in entity_values]


category_system_prompt_template = """### System:
あなたは誠実で優秀なAIアシスタントです。ユーザーの指示に可能な限り正確に従ってください。
### User:
以下の文章をカテゴリーに分類してください。カテゴリーは以下のリストに示す内のいずれかです。いずれか1つを選択して回答してください。
カテゴリーのリスト:
- AI
- リサーチ&イノベーション
- ハイブリッドクラウド
- セキュリティー
- サステナビリティー
- IBM Consulting
- 経営情報
- 導入事例
- 調査
- CSR
文章:
IBMは、年次イベント「THINK」において、Amazon Web Services（AWS）社と協力し、AIとデータのプラットフォームであるwatsonx内のIBM製品の全ポートフォリオをAWSサービスで利用可能になることを発表しました。これにより、エンドツーエンドのガバナンスに支えられた、オープンでハイブリッドなアプローチを通じ、より容易に企業におけるAIの活用を拡張できるようになります。
### Assistant:
カテゴリー:
AI
### User:
以下の文章をカテゴリーに分類してください。カテゴリーは以下のリストに示す内のいずれかです。いずれか1つを選択して回答してください。
カテゴリーのリスト:
- AI
- リサーチ&イノベーション
- ハイブリッドクラウド
- セキュリティー
- サステナビリティー
- IBM Consulting
- 経営情報
- 導入事例
- 調査
- CSR
文章:
今回の覚書では共同研究の可能性のある領域が示されており、具体的には、処理能力の飛躍的な向上と消費電力低減の両立を目指し、ブレインインスパイアードコンピューティング※やチップレットなどの半導体技術の共同研究開発を検討します。ソフトウェア技術については、ハードウェアとの協調最適化による製品の高性能化や、開発期間の短縮化を目指します。さらに、複雑化する半導体設計を適切に管理するためのオープンで柔軟なソフトウェアソリューションを検討していきます。
### Assistant:
カテゴリー:
リサーチ&イノベーション
### User:
以下の文章をカテゴリーに分類してください。カテゴリーは以下のリストに示す内のいずれかです。いずれか1つを選択して回答してください。
カテゴリーのリスト:
- AI
- リサーチ&イノベーション
- ハイブリッドクラウド
- セキュリティー
- サステナビリティー
- IBM Consulting
- 経営情報
- 導入事例
- 調査
- CSR
文章:
{context}
### Assistant:
カテゴリー:
"""
category_prompt_template = PromptTemplate(
    input_variables=[
        "context",
    ],
    template=category_system_prompt_template,
)


def categorize(summary: str):
    prompt = category_prompt_template.format(context=summary)
    response = client.text.generation.create(
        model_id=model_id, inputs=prompt, parameters=parameters
    )
    category = next(response).results[0].generated_text
    categories = ("AI", "リサーチ&イノベーション", "ハイブリッドクラウド", "セキュリティー", "サステナビリティー", "IBM Consulting", "経営情報", "導入事例", "調査", "CSR")
    return category if category in categories else "不明"


def enrich(doc):
    app.logger.info("doc: %s", doc)
    features_to_send = []
    bodies = [
        doc["artifact"][f["location"]["begin"]:f["location"]["end"]]
        for f in filter(
            lambda f: f["properties"]["field_name"] == "body", doc["features"]
        )
    ]
    chunks = create_chunks(bodies)
    summary, heading = summarize(chunks)
    category = categorize(summary)
    entities = extract_entities([summary])
    app.logger.info("summary: %s, heading: %s, category: %s, entities: %s", summary, heading, category, entities)
    features_to_send.append(
        {
            "type": "annotation",
            "properties": {
                "type": "document_classes",
                "class_name": category,
                "confidence": 1.0,
            }
        }
    )
    for feature in doc["features"]:
        try:
            if feature["properties"]["field_name"] == "title":
                # Add summary
                location = feature["location"]
                begin = location["begin"]
                end = location["end"]
                features_to_send.append(
                    {
                        "type": "annotation",
                        "location": {
                            "begin": begin,
                            "end": end,
                        },
                        "properties": {
                            "type": "entities",
                            "confidence": 1.0,
                            "entity_type": "生成AIによるタイトル",
                            "entity_text": heading,
                        },
                    }
                )
            elif feature["properties"]["field_name"] == "body":
                # Extract entities
                location = feature["location"]
                begin = location["begin"]
                end = location["end"]
                text = doc["artifact"][begin:end]
                app.logger.info("body: %s", text)
                for entity in entities:
                    entity_text = entity["text"]
                    entity_type = entity["type"]
                    for matched in re.finditer(re.escape(entity_text), text):
                        features_to_send.append(
                            {
                                "type": "annotation",
                                "location": {
                                    "begin": matched.start() + begin,
                                    "end": matched.end() + begin,
                                },
                                "properties": {
                                    "type": "entities",
                                    "confidence": 1.0,
                                    "entity_type": entity_type,
                                    "entity_text": matched.group(0),
                                },
                            }
                        )
        except Exception as e:
            # Notice example
            features_to_send.append(
                {
                    "type": "notice",
                    "properties": {
                        "description": str(e),
                        "created": round(time.time() * 1000),
                    },
                }
            )
    app.logger.info("features_to_send: %s", features_to_send)
    return {"document_id": doc["document_id"], "features": features_to_send}


def enrichment_worker():
    while True:
        item = q.get()
        version = item["version"]
        data = item["data"]
        project_id = data["project_id"]
        collection_id = data["collection_id"]
        batch_id = data["batch_id"]
        batch_api = f"{WD_API_URL}/v2/projects/{project_id}/collections/{collection_id}/batches/{batch_id}"
        params = {"version": version}
        auth = ("apikey", WD_API_KEY)
        headers = {"Accept-Encoding": "gzip"}
        try:
            # Get documents from WD
            response = requests.get(
                batch_api, params=params, auth=auth, headers=headers, stream=True
            )
            status_code = response.status_code
            app.logger.info("Pulled a batch: %s, status: %d", batch_id, status_code)
            if status_code == 200:
                # Annotate documents
                enriched_docs = [
                    enrich(json.loads(line)) for line in response.iter_lines()
                ]
                files = {
                    "file": (
                        "data.ndjson.gz",
                        gzip.compress(
                            "\n".join(
                                [
                                    json.dumps(enriched_doc)
                                    for enriched_doc in enriched_docs
                                ]
                            ).encode("utf-8")
                        ),
                        "application/x-ndjson",
                    )
                }
                # Upload annotated documents
                response = requests.post(
                    batch_api, params=params, files=files, auth=auth
                )
                status_code = response.status_code
                app.logger.info("Pushed a batch: %s, status: %d", batch_id, status_code)
        except Exception as e:
            app.logger.error("An error occurred: %s", e, exc_info=True)
            # Retry
            q.put(item)


# Turn on the enrichment worker thread
threading.Thread(target=enrichment_worker, daemon=True).start()


# Webhook endpoint
@app.route("/webhook", methods=["POST"])
def webhook():
    # Verify JWT token
    header = flask.request.headers.get("Authorization")
    _, token = header.split()
    try:
        jwt.decode(token, WEBHOOK_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as e:
        app.logger.error("Invalid token: %s", e)
        return {"status": "unauthorized"}, 401
    # Process webhook event
    data = flask.json.loads(flask.request.data)
    app.logger.info("Received event: %s", data)
    event = data["event"]
    if event == "ping":
        # Receive this event when a webhook enrichment is created
        code = 200
        status = "ok"
    elif event == "enrichment.batch.created":
        # Receive this event when a batch of the documents gets ready
        code = 202
        status = "accepted"
        # Put an enrichment request into the queue
        q.put(data)
    else:
        # Unknown event type
        code = 400
        status = "bad request"
    return {"status": status}, code


PORT = os.getenv("PORT", "8080")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(PORT))
