# Description
The technology of Retrieval-Augmented Generation (RAG) using Large Language Models (LLMs) has been gaining popularity recently. To deepen my understanding of this field, I decided to create a simple desktop application for RAG using Notion as a dataset. Notion is a tool I frequently use for taking notes and studying, making it the perfect choice for this project. <br>
Additionally, I set a restriction on not spending any money during the app development process

# History
Version | Description
--- | --- 
V1.0 | Notion API & RAG Prototype App
V1.1 | Evaluation mode using RAGAs & config.yaml

# Technologies & Env
- Python 3.11
- Notion API
- Free LLM API
- LangChain
- Streamlit (UI)
- CPU
  
# Result
### Original: 
```How much is the accommodation at Warsaw University of Technology?```
![Screen Shot 2024-12-24 at 18 18 52](https://github.com/user-attachments/assets/36fea644-8cb5-4db7-8f3f-a1f65cb6e5f3)

### After retreiving data from Notion:
![Screen Shot 2024-12-24 at 18 19 25](https://github.com/user-attachments/assets/0e81bb49-aa00-44d4-b09b-4087939f3c53)

```How much is the accommodation at Warsaw University of Technology?```
![Screen Shot 2024-12-24 at 18 21 52](https://github.com/user-attachments/assets/7a4afe36-ee11-45e7-a289-7e4b45162e6a)

**The answer is based on the content in Notion**

# Evaluation

For evaluation, I utilized RAGAS along with the Llama 3 LLM model. The evaluation was conducted using 5 questions paired with their corresponding ground truth answers.

common parameters:
```
embedding:
  model: all-mpnet-base-v2  # all-MiniLM-L12-v2, all-mpnet-base-v2, all-distilroberta-v1 etc
  chunk_size: 1024
  overlap: 100
  db_dir: ./chroma_db

llm:
  model: mistralai/Mistral-7B-Instruct-v0.2 # google/gemma-2b-it, mistralai/Mistral-7B-Instruct-v0.2, openai-community/gpt2
  search_type: mmr          
  temperature: 0.4
  k: 4
  fetch_k: 20
  max_token: 1024
```

Embedding Model | LLM Model | parameter change | Answer Relevancy | Context Recall | Human check | average answer time (s) | Note
---  |  --- | --- | --- | --- | --- | ---
all-mpnet-base-v2 | google/gemma-2b-it |  | **0.6279** | 0.7578 | 4/5 | 12.68 | This looks the best
all-mpnet-base-v2 | google/gemma-2b-it | chunk_size: 512 | 0.1460 | 0.6000 | 2/5 | 21.87
all-MiniLM-L12-v2 | google/gemma-2b-it | | 0.4875 | 0.7000 | 3/5 | 22.40
all-MiniLM-L12-v2 | mistralai/Mistral-7B-Instruct-v0.2 |  | 0.5545 | 0.6933 | 5/5 | 1.66 | The answer is correct, but repeat the same thing over and over again.
all-MiniLM-L12-v2 | openai-community/gpt2 | | 0.3485 | 0.7733 | 5/5 | 19.23 | The answer is correct, but repeat the same thing over and over again.


HuggingFace: https://huggingface.co/models




