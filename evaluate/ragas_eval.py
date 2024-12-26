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
from ragas.metrics import answer_correctness, answer_relevancy, context_recall
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.run_config import RunConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm import RAGApp


# logging setting
logging.basicConfig(
    filename="evaluate/evaluation.log",
    level=logging.DEBUG,  
    format="%(asctime)s - %(levelname)s - %(message)s",
)
os.environ["RAGAS_DEBUG"] = "true"

# Questions
questions = [
    "What phenomenon allows Silicoids to share their memories and preserve their culture?",
    "How does the surface of Planet Z create iridescent hues in the sky?",
    "What is the primary energy source for Silicoids on Planet Z?",
    "What unique material is used for constructing buildings and vehicles on Planet Z?",
    "What unsolved mystery lies deep beneath the surface of Planet Z?",

]

# Ground Truth Answers
ground_truth = [
    "The 'Winds of Memory' enable Silicoids to share their memories and preserve their culture by sweeping periodic energy waves across the planet.",
    "The translucent crystals covering the surface of Planet Z act as natural prisms, refracting light and painting the sky in iridescent hues.",
    "Silicoids harness light and vibrations as their primary energy sources.",
    "Buildings and vehicles on Planet Z are biologically 'grown' from crystals, which can communicate with intelligent lifeforms and function as tools and companions.",
    "The 'Infinite Corridor', a vast subterranean structure inscribed with undecipherable symbols believed to be remnants of an ancient intelligent species, remains an unsolved mystery.",
]

def main():
    try:
        
        # data = {
        #     "user_input": [],
        #     "response": [],
        #     "ground_truth": [],
        #     "contexts": [],
        # }

        # # Get response for each question
        # for i, question in enumerate(questions):

        #     eval_rag = RAGApp()

        #     response = eval_rag.get_response(question, eval_mode=True)
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
        # print(data)
        dataset = Dataset.from_dict(data)

        # ---------------------------------------------------------------

        eval_llm = ChatOllama(model="llama3.2", timeout=60000)
        eval_embeddings = OllamaEmbeddings(model="llama3.2")

        # Increase the timeout settings
        run_config = RunConfig(timeout=600.0, log_tenacity=True, max_workers=2)  # max_workers=2

        # list of mtrics 
        metrics = [answer_relevancy, context_recall]
        # metrics = [answer_correctness]

        results = {}

        # Evaluate one by one otherwise I get timeout error
        try:
            # iterate each metric
            for metric in metrics:
                print(f"Evaluating metric: {metric.name}") 
                logging.info(f"Evaluating metric: {metric.name}")

                # Evaluate metrics
                result = evaluate(
                    dataset=dataset,
                    metrics=[metric],  
                    llm=eval_llm,
                    embeddings=eval_embeddings,
                    run_config=run_config,
                )

                results[metric.name] = result

                # result
                print(f"{metric.name} result:", result)
                logging.info(f"{metric.name} result: {result}") 

        except Exception as e:
            print(f"Main loop error: {e}")
            logging.error(f"Evaluation error: {e}")

    except Exception as e:
        print(f"Main loop error: {e}")
        logging.error(f"Main loop error: {e}")

if __name__ == "__main__":
    main()
