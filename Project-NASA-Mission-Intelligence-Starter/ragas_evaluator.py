from typing import Dict, List, Optional

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

try:
    from ragas import SingleTurnSample
    from ragas.metrics import ResponseRelevancy, Faithfulness
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""

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
