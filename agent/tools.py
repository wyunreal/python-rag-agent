"""Tools exposed to the ReAct agent."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agent.config import Settings
from agent.index import embed_query, ensure_index, format_hits
from agent.tracing import span


def make_tools(settings: Settings) -> list[BaseTool]:
    """Create the tool set bound to one configuration snapshot."""

    @tool
    def search_corpus(
        query: str,
        k: int = settings.top_k,
        norm_id: str | None = None,
    ) -> str:
        """Busca preceptos en el corpus de legislación española consolidada (BOE).

        Devuelve los fragmentos más relevantes con su norma, artículo, fecha de
        vigencia y URL oficial. Úsala siempre que necesites el texto legal antes
        de responder.

        Args:
            query: consulta en lenguaje natural, por ejemplo "plazo de detención
                preventiva" o "requisitos para contraer matrimonio".
            k: número de fragmentos a recuperar (por defecto 5).
            norm_id: filtra por una norma concreta, por ejemplo
                "BOE-A-1978-31229" para la Constitución. Déjalo vacío para
                buscar en todo el corpus.
        """
        with span(
            "retrieval",
            input={"query": query, "k": k, "norm_id": norm_id},
        ) as retrieval:
            with span("ensure_index") as observation:
                index = ensure_index(settings)
                if observation is not None:
                    observation.update(output={"chunks": len(index.chunks)})

            with span("embed_query", input={"query": query}) as observation:
                query_vector = embed_query(settings, query)
                if observation is not None:
                    observation.update(output={"dim": int(query_vector.shape[0])})

            with span(
                "vector_search",
                input={"k": k, "norm_id": norm_id},
            ) as observation:
                hits = index.search(query_vector, k=k, norm_id=norm_id)
                if observation is not None:
                    observation.update(
                        output={
                            "hits": [
                                {
                                    "id": hit.chunk.id,
                                    "label": hit.chunk.label(),
                                    "score": round(hit.score, 4),
                                }
                                for hit in hits
                            ]
                        }
                    )

            with span("format_hits") as observation:
                output = format_hits(hits)
                if observation is not None:
                    observation.update(output={"chars": len(output), "hit_count": len(hits)})

            if retrieval is not None:
                retrieval.update(
                    output={
                        "hits": [hit.chunk.id for hit in hits],
                        "chars": len(output),
                    }
                )
            return output

    return [search_corpus]
