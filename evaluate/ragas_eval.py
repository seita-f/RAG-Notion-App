import sys
import os
import json
import time
import numpy as np
import yaml

from datasets import Dataset
import logging
from datetime import datetime
from dotenv import load_dotenv

# langchain
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint

# RAGAS
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.run_config import RunConfig

# import my class 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm import RAGApp
from embedding import Embedding, JSONHandler


# logging setting
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"evaluate/evaluation_{current_time}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,  
    format="%(asctime)s - %(levelname)s - %(message)s",
)
os.environ["RAGAS_DEBUG"] = "true"

# Questions
questions = [
    "Why did the villagers avoid the Lake of Stars?",
    "What motivated Emma to visit the Lake of Stars?",
    "What were the three trials Emma had to overcome?",
    "How did the Spirit reward Emma after completing the trials?",
    "What change occurred in the villagers' perception of the lake after Emma's visits?",
]

# Ground Truth Answers
ground_truth = [
    "The villagers believed that the Spirit of the Stars resided there and would test anyone who approached the lake",
    "Emma was fascinated by the stories of the lake's beauty and wanted to see its brilliance with her own eyes.",
    "First, finding the 'Eternal Star Fragment' hidden at the lake's edge. Second, helping the forest animals by resolving their troubles. Third, overcoming her inner fears to see the lake's true form.",
    "The Spirit entrusted the lake's secret to Emma, allowing its starlight to merge with her heart.",
    "The villagers learned to appreciate the lake's beauty and value, quietly admiring its glow while respecting its sacred nature.",
]

def embedding_process(embedding_model_name: str, 
                      persist_directory: str, 
                      json_path: str, 
                      chunk_size: int, 
                      chunk_overlap: int):
    """
    Embedding 
    """
    # Load and process the JSON data
    json_data = JSONHandler.load_json_data(json_path)
    documents = JSONHandler.create_documents_from_json(json_data)

    # Create EmbeddingHandler instance and process the documents
    embedding_handler = Embedding(
        embedding_model_name=embedding_model_name,
        persist_directory=persist_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    cleaned_docs = embedding_handler.split_and_clean_documents(documents)
    embedding_handler.create_and_persist_chroma_db(cleaned_docs)

def generate_answer(questions: list, 
                    embedding_model: str,
                    llm_model: str,
                    max_token: int,
                    temperature: int,
                    search_type: int,
                    k: int,
                    fetch_k: int) -> Dataset: 
    """
    Generate answer using LLM
    """
    try:
        eval_rag = RAGApp(embedding_model=embedding_model, llm_model=llm_model, max_token=max_token, temperature=temperature)
        data = {
            "user_input": [],
            "response": [],
            "ground_truth": [],
            "contexts": [],
        }
        mesure_time = []
        # Generate answer & measure time
        for i, question in enumerate(questions):
            
            start = time.time()
            response = eval_rag.get_response(question, search_type=search_type, k=k, fetch_k=fetch_k, eval_mode=True)
            end = time.time()
            mesure_time.append(end - start)

            data["user_input"].append(response["user_input"])
            data["response"].append(response["response"])
            data["contexts"].append(response["contexts"])
            data["ground_truth"].append(ground_truth[i])  

        dataset = Dataset.from_dict(data)
        mean_time = np.mean(mesure_time)

        logging.info(f"Time: {mesure_time}")
        logging.info(f"Average time: {mean_time} ") 
        logging.info(f"user_input: {dataset['user_input']}") 
        logging.info(f"response: {dataset['response']}") 
        logging.info(f"contexts: {dataset['contexts']}") 
        logging.info(f"ground truth: {dataset['ground_truth']}")         

        return dataset
    
    except Exception as e:
        print(f"Main loop error: {e}")
        logging.error(f"Main loop error: {e}")


def main():

    # list of mtrics 
    metrics = [answer_relevancy, context_recall]
    results = {}

    # load config
    with open('config.yaml') as file:
        config = yaml.safe_load(file.read())

    logging.info(f"config={config}")

    # embedding and generate answer for evaluation      
    embedding_process(config["embedding"]["model"], 
                      config["embedding"]["db_dir"], 
                      config["notion"]["content_file"], 
                      config["embedding"]["chunk_size"], 
                      config["embedding"]["overlap"])  
    
    print("Embedding is done")
    
    dataset = generate_answer(questions, 
                              config["embedding"]["model"], 
                              config["llm"]["model"], 
                              config["llm"]["max_token"], 
                              config["llm"]["temperature"], 
                              config["llm"]["search_type"], 
                              config["llm"]["k"], 
                              config["llm"]["fetch_k"])
    
    print("Generated answers")
    
    # Evaluation
    try:           
        eval_llm = ChatOllama(model="llama3.2", timeout=60000)
        eval_embeddings = OllamaEmbeddings(model="llama3.2")
        run_config = RunConfig(timeout=config["evaluation"]["timeout"], max_workers=config["evaluation"]["max_workers"], log_tenacity=True)  

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
