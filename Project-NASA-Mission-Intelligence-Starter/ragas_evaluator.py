from typing import Dict, List
import os
import json

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from ragas import SingleTurnSample
    from ragas.metrics import ResponseRelevancy, Faithfulness
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate one RAG answer using RAGAS metrics."""

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    if not question or not answer or not contexts:
        return {"error": "Question, answer, and contexts are required"}

    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        )

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        metrics = [
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            Faithfulness(llm=evaluator_llm)
        ]

        results = {}

        for metric in metrics:
            score = metric.single_turn_score(sample)
            results[metric.name] = float(score)

        return results

    except Exception as e:
        return {"error": str(e)}


def load_evaluation_questions(file_path: str = "evaluation_dataset.txt") -> List[str]:
    """Load evaluation questions from evaluation_dataset.txt."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {file_path}")

    questions = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            question = line.strip()

            if question:
                questions.append(question)

    if len(questions) < 5:
        raise ValueError("Evaluation dataset must contain at least 5 questions.")

    return questions


def run_batch_evaluation(
    collection,
    openai_key: str,
    rag_client,
    llm_client,
    dataset_path: str = "evaluation_dataset.txt",
    n_results: int = 3,
    model: str = "gpt-3.5-turbo"
) -> Dict:
    """
    Run end-to-end batch evaluation:
    load questions -> retrieve context -> generate answer -> evaluate metrics.
    """

    questions = load_evaluation_questions(dataset_path)
    per_question_results = []

    aggregate_scores = {
        "response_relevancy": [],
        "faithfulness": []
    }

    for question in questions:
        docs_result = rag_client.retrieve_documents(
            collection,
            question,
            n_results=n_results
        )

        contexts = []
        context_text = ""

        if docs_result and docs_result.get("documents"):
            contexts = docs_result["documents"][0]
            metadatas = docs_result.get("metadatas", [[]])[0]
            context_text = rag_client.format_context(contexts, metadatas)

        answer = llm_client.generate_response(
            openai_key=openai_key,
            user_message=question,
            context=context_text,
            conversation_history=[],
            model=model
        )

        scores = evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=contexts
        )

        result = {
            "question": question,
            "answer": answer,
            "scores": scores
        }

        per_question_results.append(result)

        for metric_name, score in scores.items():
            if isinstance(score, (int, float)):
                aggregate_scores.setdefault(metric_name, []).append(score)

    aggregate_summary = {}

    for metric_name, values in aggregate_scores.items():
        if values:
            aggregate_summary[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }

    report = {
        "dataset_path": dataset_path,
        "total_questions": len(questions),
        "per_question_results": per_question_results,
        "aggregate_summary": aggregate_summary
    }

    with open("evaluation_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report
