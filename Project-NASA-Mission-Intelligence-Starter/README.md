# 🚀 NASA Mission Intelligence RAG System

## Project Overview

This project implements a complete Retrieval-Augmented Generation (RAG) system for answering questions about historic NASA missions including:

- Apollo 11
- Apollo 13
- Challenger

The system processes NASA mission documents, stores embeddings in ChromaDB, retrieves relevant context using semantic search, generates grounded responses using OpenAI models, and evaluates answer quality using RAGAS metrics.

---

# Architecture

User Question
↓
RAG Retrieval (ChromaDB)
↓
Relevant NASA Documents
↓
OpenAI LLM
↓
Generated Answer
↓
RAGAS Evaluation
↓
Faithfulness + Response Relevancy Scores

---

# Project Structure

```text
Project-NASA-Mission-Intelligence-Starter/
│
├── data_text/
│   ├── apollo11/
│   ├── apollo13/
│   └── challenger/
│
├── embedding_pipeline.py
├── rag_client.py
├── llm_client.py
├── ragas_evaluator.py
├── chat.py
├── evaluation_dataset.txt
├── requirements.txt
└── README.md
```

---

# Features

## Embedding Pipeline

- OpenAI embedding generation
- Configurable chunk size
- Configurable chunk overlap
- Metadata extraction
- ChromaDB storage
- Batch processing
- Skip / Update / Replace modes

## Retrieval System

- Semantic search
- Top-k retrieval
- Mission-specific filtering
- Context construction
- Source attribution

## LLM Integration

- OpenAI GPT models
- NASA expert system prompt
- Conversation history support
- Grounded response generation
- Hallucination reduction

## Evaluation

Uses RAGAS metrics:

- Faithfulness
- Response Relevancy

## User Interface

- Streamlit Chat UI
- OpenAI model selection
- Backend selection
- Real-time evaluation display

---
## Batch Evaluation

The system uses evaluation_dataset.txt to run automated end-to-end testing.

Workflow:
1. Load evaluation questions
2. Retrieve relevant documents from ChromaDB
3. Generate responses using OpenAI
4. Evaluate responses using RAGAS
5. Save results to evaluation_report.json

Metrics:
- Faithfulness
- Response Relevancy

Aggregate statistics such as average scores are generated automatically.
# Installation

## Clone Repository

```bash
git clone <repository-url>
cd Project-NASA-Mission-Intelligence-Starter
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

```text
chromadb==1.5.7
openai==2.31.0
langchain==0.3.26
langchain-openai==1.1.13
ragas==0.4.3
pandas==3.0.2
streamlit==1.56.0
```

---

# Configure OpenAI API Key

Linux/Mac:

```bash
export OPENAI_API_KEY="your-api-key"
```

Windows CMD:

```cmd
set OPENAI_API_KEY=your-api-key
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

---

# Building the Vector Database

Run the embedding pipeline:

```bash
python embedding_pipeline.py \
--openai-key YOUR_KEY \
--data-path ./data_text \
--chroma-dir ./chroma_db_openai \
--collection-name nasa_space_missions_text \
--chunk-size 500 \
--chunk-overlap 100 \
--update-mode replace
```

This will:

1. Read NASA text files
2. Split documents into chunks
3. Generate OpenAI embeddings
4. Store vectors in ChromaDB
5. Save metadata including:
   - Mission
   - Source file
   - Chunk information

---

# View Collection Statistics

```bash
python embedding_pipeline.py \
--openai-key YOUR_KEY \
--chroma-dir ./chroma_db_openai \
--collection-name nasa_space_missions_text \
--stats-only
```

Example output:

```text
Collection Name: nasa_space_missions_text
Total Documents: 1250
Missions:
Apollo 11
Apollo 13
Challenger
```

---

# Running the Chat Application

Launch Streamlit:

```bash
streamlit run chat.py
```

Open browser:

```text
http://localhost:8501
```

---

# Example Questions

## Apollo 11

- What was the primary objective of Apollo 11?
- Who were the crew members of Apollo 11?
- Describe the lunar landing sequence.

## Apollo 13

- What technical problems occurred during Apollo 13?
- Why is Apollo 13 considered a successful failure?
- How did the crew survive the oxygen tank explosion?

## Challenger

- What caused the Challenger disaster?
- What recommendations followed the investigation?
- What lessons were learned from Challenger?

---

# Evaluation Dataset

The project includes:

```text
evaluation_dataset.txt
```

Sample evaluation questions:

1. What was the primary objective of Apollo 11?
2. Who were the crew members of Apollo 11?
3. What technical problems occurred during Apollo 13?
4. Why is Apollo 13 considered a successful failure?
5. What events led to the Challenger disaster?
6. What recommendations followed the Challenger investigation?
7. Summarize the timeline of Apollo 11 from launch to landing.

---

# RAGAS Evaluation

For every generated answer the system computes:

## Faithfulness

Measures whether the response is supported by the retrieved context.

Range:

```text
0.0 → 1.0
```

Higher is better.

---

## Response Relevancy

Measures how well the answer addresses the user's question.

Range:

```text
0.0 → 1.0
```

Higher is better.

---
## Dataset-Based Batch Evaluation

The project includes `evaluation_dataset.txt`, which contains mission-relevant questions across overview, crew, technical, emergency, disaster-analysis, and timeline categories.

The dataset is used by the standalone batch evaluation workflow in `batch_evaluate.py`.

Run batch evaluation:

```bash
python batch_evaluate.py \
  --openai-key YOUR_KEY \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --dataset-path evaluation_dataset.txt \
  --output-path evaluation_report.json

# Metadata Stored in ChromaDB

Each chunk stores:

```json
{
  "mission": "apollo11",
  "source": "apollo11_transcript.txt",
  "chunk_index": 5,
  "chunk_start": 2500,
  "chunk_end": 3000,
  "chunk_size": 500
}
```

---

# Technologies Used

- Python
- OpenAI API
- ChromaDB
- Streamlit
- LangChain
- RAGAS
- Pandas

---

# Future Improvements

- Hybrid Search (BM25 + Vector Search)
- Citation highlighting
- Multi-modal NASA document support
- Additional RAGAS metrics
- Conversation memory persistence
- Azure OpenAI support

---

# Author

Alankriti Mehra

MCA Student, PES University  
Data Engineering • Generative AI • Cloud Computing

---

# License

This project is developed for educational purposes as part of the Udacity Generative AI Nanodegree program.
