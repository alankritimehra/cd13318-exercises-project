import chromadb
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    chroma_dirs = [
        path for path in current_dir.iterdir()
        if path.is_dir() and "chroma" in path.name.lower()
    ]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collections = client.list_collections()

            for collection in collections:
                key = f"{chroma_dir.name}:{collection.name}"

                try:
                    count = collection.count()
                except Exception:
                    count = 0

                backends[key] = {
                    "path": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{collection.name} ({chroma_dir}) - {count} docs",
                    "document_count": str(count)
                }

        except Exception as e:
            key = f"{chroma_dir.name}:error"
            backends[key] = {
                "path": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} - Error: {str(e)[:60]}",
                "document_count": "0"
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend"""
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(name=collection_name)
    return collection


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None

    if mission_filter and mission_filter.lower() not in ["all", "all missions", "none"]:
        where_filter = {"mission": mission_filter}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        return results

    except Exception as e:
        print(f"Retrieval error: {e}")
        return None


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts = ["Retrieved NASA Mission Context:\n"]

    seen_docs = set()

    for i, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        if not doc or doc in seen_docs:
            continue

        seen_docs.add(doc)

        mission = metadata.get("mission", "unknown").replace("_", " ").title()
        source = metadata.get("source", "unknown")
        category = metadata.get("document_category", "general").replace("_", " ").title()

        header = f"\n--- Source {i}: Mission={mission}, Source={source}, Category={category} ---"
        context_parts.append(header)

        if len(doc) > 1500:
            doc = doc[:1500] + "..."

        context_parts.append(doc)

    return "\n".join(context_parts)
