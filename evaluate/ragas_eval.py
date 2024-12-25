import sys
import os

from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceEndpoint
import logging

# For Evaluate
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm import RAGApp

# Questions
questions = [
    "What is the average temperature on Planet Z?",
    "What is the primary energy source for Silicoids?",
    "How far is Planet Z from Earth?"
]

# Ground Truth Answers
ground_truth = [
    "The average temperature on Planet Z is 30°C during the day and 20°C at night.",
    "Silicoids harness light and vibrations as their primary energy sources.",
    "Planet Z is approximately 450 light-years away from Earth."
]

# class EvalRAG(RAGApp):
#     def get_response(self, question):
#         rag_chain = (
#             RunnablePassthrough()  # Pass the formatted string directly
#             | self.llm  # The HuggingFaceEndpoint processes the input string
#         )

#         retriever = self.db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
#         docs = retriever.invoke(question)

#         # Format context as a list of strings
#         formatted_context = [doc.page_content for doc in docs]

#         formatted_input = self.__format_rag_input__(
#             context=" ".join(formatted_context), question=question
#         )
#         result = rag_chain.invoke(formatted_input)

#         return {
#             "user_input": question,
#             "contexts": formatted_context,  # Pass as a list of strings
#             "response": result,
#             "ground_truth": ground_truth,

#         }

# def main():
    
#     # モデル統合
#     try:
        
#         contexts = []
#         answers = []
        
#         eval_rag = EvalRAG()
#         data = eval_rag.get_response(questions)

#         print(data)

#         # # llm = HuggingFaceEndpoint(repo_id="google/gemma-2b-it", max_length=1024, temperature=0.4)
#         # # embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
#         # eval_llm = ChatOllama(model="llama3.2", timeout=2000)
#         # eval_embeddings = OllamaEmbeddings(model="llama3.2")

#         # data = {
#         #     "user_input": [
#         #         "What is the capital of France?",
#         #         "Explain the process of photosynthesis."
#         #     ],
#         #     "response": [
#         #         "Paris",
#         #         "Photosynthesis"
#         #     ],
#         #     "ground_truth": [
#         #         "Paris",
#         #         "Photosynthesis"
#         #     ],
#         #     "contexts": [
#         #         ["Paris is the capital of France."],
#         #         ["Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll."]
#         #     ]
#         # }

#         # # Datasetを作成
#         # dataset = Dataset.from_dict(data)

#         # # Prepare dataset for evaluation
#         # ds = Dataset.from_dict(
#         #     {
#         #         "question": questions,
#         #         "answer": answers,
#         #         "contexts": contexts,  # Must be a list of strings for each question
#         #         "ground_truth": ground_truth,
#         #     }
#         # )
#         # dataset = get_dataset()

#         # print(data)
#         # print()
#         # print(data["eval"]["question"])
#         # print()
#         # print(data["eval"]["answer"])
#         # print()
#         # print(data["eval"]["ground_truth"])
#         # print()
#         # print(data["eval"]["contexts"])

#         # try:
#         #     result_relevancy = evaluate(
#         #         dataset=dataset,
#         #         metrics=[answer_relevancy],
#         #         llm=eval_llm,
#         #         embeddings=eval_embeddings,
#         #     )
#         #     print("Answer Relevancy result:", result_relevancy)
#         # except Exception as e:
#         #     print(f"Answer relevancy evaluation error: {e}")

    
#         # print("\nEvaluation Results:", result)

#     except Exception as e:
#         logging.error(f"Main loop error: {e}")

# if __name__ == "__main__":
#     main()

class EvalRAG(RAGApp):
    def get_response(self, question):
        rag_chain = (
            RunnablePassthrough()  # Pass the formatted string directly
            | self.llm  # The HuggingFaceEndpoint processes the input string
        )

        retriever = self.db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
        docs = retriever.invoke(question)

        # Format context as a list of strings
        formatted_context = [doc.page_content for doc in docs]

        formatted_input = self.__format_rag_input__(
            context=" ".join(formatted_context), question=question
        )
        result = rag_chain.invoke(formatted_input)

        return {
            "user_input": question,
            "contexts": formatted_context,  # Pass as a list of strings
            "response": result,
        }

def main():
    try:
        # モデル統合
        eval_rag = EvalRAG()

        # データ構造の初期化
        data = {
            "user_input": [],
            "response": [],
            "ground_truth": [],
            "contexts": [],
        }

        # 各質問に対して get_response を呼び出す
        for i, question in enumerate(questions):
            response = eval_rag.get_response(question)

            # データ構造に追加
            data["user_input"].append(response["user_input"])
            data["response"].append(response["response"])
            data["contexts"].append(response["contexts"])
            data["ground_truth"].append(ground_truth[i])  # 対応する ground_truth を追加

        # データセットの作成
        dataset = Dataset.from_dict(data)

        print(dataset)

        print(dataset["user_input"])
        print()
        print(dataset["response"])
        print()
        print(dataset["contexts"])
        print()
        print(dataset["ground_truth"])

        eval_llm = ChatOllama(model="llama3.2", timeout=10000)
        eval_embeddings = OllamaEmbeddings(model="llama3.2")

        try:
            result_relevancy = evaluate(
                dataset=dataset,
                metrics=[answer_relevancy],
                llm=eval_llm,
                embeddings=eval_embeddings,
            )
            
            print("Answer Relevancy result:", result_relevancy)
        except Exception as e:
            print(f"Answer relevancy evaluation error: {e}")

    except Exception as e:
        logging.error(f"Main loop error: {e}")

if __name__ == "__main__":
    main()
