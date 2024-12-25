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
        # model
        eval_rag = EvalRAG()

        data = {
            "user_input": [],
            "response": [],
            "ground_truth": [],
            "contexts": [],
        }

        # Get response for each question
        for i, question in enumerate(questions):
            response = eval_rag.get_response(question)

            data["user_input"].append(response["user_input"])
            data["response"].append(response["response"])
            data["contexts"].append(response["contexts"])
            data["ground_truth"].append(ground_truth[i])  # 対応する ground_truth を追加

        dataset = Dataset.from_dict(data)

        # DEBUG:
        # print(dataset)
        # print(dataset["user_input"])
        # print()
        # print(dataset["response"])
        # print()
        # print(dataset["contexts"])
        # print()
        # print(dataset["ground_truth"])

        eval_llm = ChatOllama(model="llama3.2", timeout=30000)
        eval_embeddings = OllamaEmbeddings(model="llama3.2")

        # 使用するメトリクスのリスト
        metrics = [answer_relevancy, faithfulness, context_precision, context_recall]

        # 評価結果を格納する辞書
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
