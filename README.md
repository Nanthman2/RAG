# Assistant Supply Chain — RAG local

Assistant qui répond à des questions logistiques (Incoterms, transport, stocks, retours, KPI, douane) en s'appuyant **uniquement** sur un corpus de documents locaux, avec citation systématique des sources et refus explicite ("je ne sais pas") quand l'information n'est pas dans le corpus.

Étape 1 du projet complet (RAG + agent SQL) : ce dépôt couvre le **RAG documentaire**. L'agent SQL sur commandes/expéditions/retours viendra dans une étape suivante, avec un routeur RAG ↔ SQL.

## Architecture

```
Question utilisateur
        │
        ▼
  Embedding (nomic-embed-text, Ollama)
        │
        ▼
  Recherche vectorielle (numpy, similarité cosinus, top-k)
        │
        ▼
  Chunks pertinents + score de similarité
        │
   (si rien au-dessus du seuil → "je ne sais pas")
        │
        ▼
  Prompt avec contexte + question (Ollama, llama3.2)
        │
        ▼
  Réponse avec citation des sources
```

## Stack

- **LLM & embeddings** : [Ollama](https://ollama.com) en local — `llama3.2` (génération) + `nomic-embed-text` (embeddings). Aucune clé API, aucun coût, tout tourne sur la machine.
- **Vector store** : index maison en numpy (embeddings + similarité cosinus en force brute), persisté dans `data/vector_store.pkl`. Choix délibéré plutôt que Chroma/FAISS : à l'échelle d'une poignée de documents ça reste instantané, ça évite toute dépendance native (Chroma/FAISS nécessitent un compilateur C++ sur Windows), et ça montre la mécanique de la recherche vectorielle plutôt que de la cacher derrière une lib.
- **UI** : [Streamlit](https://streamlit.io/).
- Pas de framework RAG (LangChain/LlamaIndex) : le pipeline (chunking, embeddings, retrieval, prompt) est écrit à la main dans `rag/` pour bien maîtriser chaque étape.

## Installation

```bash
pip install -r requirements.txt

# Modèles Ollama nécessaires (une seule fois)
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Utilisation

```bash
# 1. Indexer les documents de data/docs/
python ingest.py

# 2. Lancer l'interface
streamlit run app.py
```

## Structure du projet

```
data/docs/            Documents source (Incoterms, glossaire, transport, retours, stocks, douane, KPI, emballage)
data/vector_store.pkl Index vectoriel persisté (généré par ingest.py)
rag/chunking.py      Découpage en chunks avec chevauchement, au niveau des paragraphes
rag/ollama_client.py Wrapper embeddings + génération via Ollama
rag/vectorstore.py   Index vectoriel maison (numpy, similarité cosinus)
rag/rag_chain.py      Chaîne complète : retrieval + prompt anti-hallucination + génération
ingest.py            Script d'indexation des documents
app.py               Interface Streamlit (chat + affichage des sources)
config.py            Paramètres (modèles, taille de chunk, seuil de similarité, top-k)
```

## Garde-fous anti-hallucination

- Le prompt impose de répondre **uniquement** à partir du contexte fourni, avec citation de la source.
- Si aucun chunk récupéré ne dépasse le seuil de similarité (`MIN_SIMILARITY` dans `config.py`), l'assistant répond explicitement qu'il ne sait pas, **sans appeler le LLM** — évite qu'un LLM comble les trous par hallucination.
- Les sources et leur score de similarité sont affichés dans l'UI pour audit.

## Ce que j'ai appris

*(à compléter au fil du projet : stratégie de chunking, éval de la récupération, réduction des hallucinations, comparaison local vs API)*

## Prochaines étapes

- Agent SQL sur une base synthétique commandes/expéditions/retours (DuckDB/SQLite + Faker).
- Routeur RAG ↔ SQL selon le type de question.
- Jeu d'évaluation (~20 questions/réponses) pour mesurer la justesse et le taux de récupération correct.
- Logging des requêtes, chunks récupérés, latence.
