import sys
import os
import json

from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceEndpoint

from datasets import Dataset
import logging

# RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.run_config import RunConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm import RAGApp

# Questions
questions = [
    "Does Planet Z experience 5 hours of daylight and 5 hours of night in a single day? Please answer with either 'Yes' or 'No'. Do not provide any additional explanation.",
    "Are Silicoids carbon-based organisms? Please answer with either 'Yes' or 'No'. Do not provide any additional explanation.",
    "Can the Winds of Memory preserve and share the memories of Silicoids? Please answer with either 'Yes' or 'No'. Do not provide any additional explanation.",
    "Are buildings and vehicles on Planet Z constructed using traditional materials like metals and concrete? Please answer with either 'Yes' or 'No'. Do not provide any additional explanation.",
    "Is the Infinite Corridor located on the surface of Planet Z? Please answer with either 'Yes' or 'No'. Do not provide any additional explanation.",
]

# Ground Truth Answers
ground_truth = [
    "Yes",
    "No",
    "Yes",
    "No",
    "No",
]


class EvalRAG(RAGApp):
    def get_response(self, question):
        rag_chain = (
            RunnablePassthrough()  # Pass the formatted string directly
            | self.llm  # The HuggingFaceEndpoint processes the input string
        )

        retriever = self.db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
        docs = retriever.invoke(question)

        # Select only the first context
        formatted_context = [docs[0].page_content] if docs else []

        formatted_input = self.__format_rag_input__(
            context=" ".join(formatted_context), question=question
        )
        result = rag_chain.invoke(formatted_input)

        return {
            "user_input": question,
            "contexts": formatted_context,  # Pass the first context
            "response": result,
        }

def main():
    try:
        # model
        # eval_rag = EvalRAG()

        # data = {
        #     "user_input": [],
        #     "response": [],
        #     "ground_truth": [],
        #     "contexts": [],
        # }

        # # Get response for each question
        # for i, question in enumerate(questions):
        #     response = eval_rag.get_response(question)

        #     data["user_input"].append(response["user_input"])
        #     data["response"].append(response["response"])
        #     data["contexts"].append(response["contexts"])
        #     data["ground_truth"].append(ground_truth[i])  

        # # データを保存するファイル名
        # data_file = "evaluate/response_data.json"

        # # JSON形式でデータを保存
        # with open(data_file, "w") as f:
        #     json.dump(data, f, indent=4)

        # print(f"Data has been saved to {data_file}")

        #---------------------------------------------------------------

        # 保存されたデータを読み込む
        data_file = "evaluate/response_data.json"
        with open(data_file, "r") as f:
            data = json.load(f)

        print("Data has been loaded:")
        print(data)
        dataset = Dataset.from_dict(data)

        # data = {
        #     "user_input": [
        #         "What is the capital of France?",
        #         "Explain the process of photosynthesis."
        #     ],
        #     "response": [
        #         "Paris",
        #         "Photosynthesis"
        #     ],
        #     "ground_truth": [
        #         "Paris",
        #         "Photosynthesis"
        #     ],
        #     "contexts": [
        #         ["Paris is the capital of France."],
        #         ["Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll."]
        #     ]
        # }

        # # Datasetを作成
        # dataset = Dataset.from_dict(data)

        # # DEBUG:
        # print(dataset)
        # # print(dataset["user_input"])
        # print()
        # print(dataset["response"])
        # print()
        # # print(dataset["contexts"])
        # print()
        # print(dataset["ground_truth"])

        eval_llm = ChatOllama(model="llama3.2", timeout=15000)
        eval_embeddings = OllamaEmbeddings(model="llama3.2")

        # Increase the timeout settings
        run_config = RunConfig(timeout=300.0, max_workers=2)  # Increase timeout to 120 seconds

        # list of mtrics 
        # metrics = [answer_relevancy, faithfulness, context_precision, context_recall]
        metrics = [faithfulness]

        results = {}

        # Evaluate one by one otherwise I get timeout error
        try:
            # 各メトリクスをループ処理
            for metric in metrics:
                print(f"Evaluating metric: {metric.name}") 
                
                # メトリクスを評価
                result = evaluate(
                    dataset=dataset,
                    metrics=[metric],  
                    llm=eval_llm,
                    embeddings=eval_embeddings,
                    run_config=run_config,
                )

                results[metric.name] = result

                # 結果を表示
                print(f"{metric.name} result:", result)

        except Exception as e:
            print(f"Evaluation error: {e}")


    except Exception as e:
        logging.error(f"Main loop error: {e}")

if __name__ == "__main__":
    main()
