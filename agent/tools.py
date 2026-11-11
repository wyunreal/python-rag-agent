"""Tools exposed to the ReAct agent."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agent.config import Settings
from agent.index import ensure_index, format_hits, get_embedder


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
        index = ensure_index(settings)
        query_vector = get_embedder(settings).encode([query])[0]
        hits = index.search(query_vector, k=k, norm_id=norm_id)
        return format_hits(hits)

    return [search_corpus]
