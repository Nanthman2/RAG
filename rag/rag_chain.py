"""Chaîne RAG : retrieval vectoriel + génération avec citations, ou refus si hors contexte."""

from dataclasses import dataclass

from config import MIN_SIMILARITY, TOP_K
from rag.ollama_client import embed, generate
from rag.vectorstore import get_store

PROMPT_TEMPLATE = """Tu es un assistant spécialisé en logistique et supply chain. \
Réponds à la question UNIQUEMENT à partir du contexte fourni ci-dessous. \
Si le contexte ne contient pas l'information nécessaire pour répondre, réponds exactement : \
"Je ne sais pas, cette information n'est pas présente dans les documents fournis." \
Ne fais aucune supposition en dehors du contexte. Si le contexte développe un acronyme ou un sigle \
(par exemple FOB, CIF, OTIF), recopie EXACTEMENT ce développement tel qu'il apparaît dans le contexte, \
sans l'inventer ni le reformuler à partir de tes propres connaissances. Cite le nom du document source \
entre parenthèses après chaque affirmation, par exemple (source: incoterms_2020.md).

Contexte :
{context}

Question : {question}

Réponse :"""


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict]
    grounded: bool  # False si aucun chunk pertinent n'a été trouvé


def answer_question(question: str) -> RagAnswer:
    store = get_store()
    question_embedding = embed(question)
    hits = store.query(question_embedding, TOP_K)

    relevant_hits = [h for h in hits if h["similarity"] >= MIN_SIMILARITY]

    if not relevant_hits:
        return RagAnswer(
            answer="Je ne sais pas, cette information n'est pas présente dans les documents fournis.",
            sources=[],
            grounded=False,
        )

    context = "\n\n---\n\n".join(
        f"[Source: {h['source']}]\n{h['text']}" for h in relevant_hits
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = generate(prompt)

    return RagAnswer(answer=answer, sources=relevant_hits, grounded=True)
