import os
import json
import argparse

import rag_client
import llm_client
import ragas_evaluator


def load_questions(dataset_path: str):
    """Load questions from evaluation_dataset.txt."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    questions = []

    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            question = line.strip()
            if question:
                questions.append(question)

    if len(questions) < 5:
        raise ValueError("evaluation_dataset.txt must contain at least 5 questions.")

    return questions


def calculate_aggregate(results):
    """Calculate aggregate metric statistics."""
    metric_values = {}

    for item in results:
        scores = item.get("scores", {})

        for metric_name, value in scores.items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(metric_name, []).append(value)

    aggregate = {}

    for metric_name, values in metric_values.items():
        aggregate[metric_name] = {
            "average": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
            "count": len(values)
        }

    return aggregate


def run_batch_evaluation(
    openai_key: str,
    chroma_dir: str,
    collection_name: str,
    dataset_path: str,
    output_path: str,
    n_results: int,
    model: str
):
    """Run retrieval -> generation -> evaluation for every dataset question."""

    questions = load_questions(dataset_path)

    collection = rag_client.initialize_rag_system(
        chroma_dir=chroma_dir,
        collection_name=collection_name
    )

    per_question_results = []

    for index, question in enumerate(questions, start=1):
        print(f"Evaluating question {index}/{len(questions)}: {question}")

        docs_result = rag_client.retrieve_documents(
            collection=collection,
            query=question,
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

        scores = ragas_evaluator.evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=contexts
        )

        per_question_results.append({
            "question": question,
            "answer": answer,
            "retrieved_context_count": len(contexts),
            "scores": scores
        })

    aggregate_summary = calculate_aggregate(per_question_results)

    report = {
        "dataset_path": dataset_path,
        "total_questions": len(questions),
        "per_question_results": per_question_results,
        "aggregate_summary": aggregate_summary
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nBatch evaluation complete.")
    print(f"Report saved to: {output_path}")
    print("\nAggregate Summary:")
    print(json.dumps(aggregate_summary, indent=2))

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run batch evaluation for NASA RAG system using evaluation_dataset.txt"
    )

    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="Chroma collection name")
    parser.add_argument("--dataset-path", default="evaluation_dataset.txt", help="Evaluation dataset path")
    parser.add_argument("--output-path", default="evaluation_report.json", help="Output report path")
    parser.add_argument("--n-results", type=int, default=3, help="Number of retrieved chunks")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI chat model")

    args = parser.parse_args()

    if not args.openai_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass --openai-key.")

    run_batch_evaluation(
        openai_key=args.openai_key,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        n_results=args.n_results,
        model=args.model
    )


if __name__ == "__main__":
    main()
