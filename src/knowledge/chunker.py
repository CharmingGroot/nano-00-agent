"""Recursive text chunker with overlap for knowledge embedding."""
import logging
from typing import Any

from src.middleware.token_counter import TokenCounter
from config.settings import settings

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits text into token-bounded chunks with overlap.

    Default: 512 tokens per chunk, 50 token overlap.
    Splits on paragraph > sentence > word boundaries.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size_tokens,
        overlap: int = settings.chunk_overlap_tokens,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Chunk all pages of a parsed document.

        Args:
            pages: Output from DocumentParser.parse()

        Returns:
            List of {"text": str, "token_count": int, "metadata": dict, "chunk_index": int}
        """
        all_chunks = []
        chunk_index = 0

        for page in pages:
            text = page["text"]
            page_meta = page["metadata"]
            chunks = self._split_text(text)

            for chunk_text in chunks:
                token_count = TokenCounter.count_tokens(chunk_text)
                all_chunks.append({
                    "text": chunk_text,
                    "token_count": token_count,
                    "metadata": {**page_meta, "chunk_index": chunk_index},
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

        logger.info("Chunked document into %d chunks", len(all_chunks))
        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks respecting token limits with overlap."""
        if TokenCounter.count_tokens(text) <= self.chunk_size:
            return [text]

        # Try splitting by paragraphs first
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            return self._merge_splits(paragraphs)

        # Fall back to sentences
        sentences = self._split_sentences(text)
        if len(sentences) > 1:
            return self._merge_splits(sentences)

        # Last resort: split by words
        words = text.split()
        return self._merge_splits(words, join_char=" ")

    def _merge_splits(self, pieces: list[str], join_char: str = "\n\n") -> list[str]:
        """Merge small pieces into chunks up to chunk_size, with overlap."""
        chunks = []
        current_pieces: list[str] = []
        current_tokens = 0

        for piece in pieces:
            piece_tokens = TokenCounter.count_tokens(piece)

            if current_tokens + piece_tokens > self.chunk_size and current_pieces:
                # Emit current chunk
                chunk_text = join_char.join(current_pieces)
                chunks.append(chunk_text)

                # Keep overlap: take pieces from end that fit in overlap budget
                overlap_pieces: list[str] = []
                overlap_tokens = 0
                for p in reversed(current_pieces):
                    pt = TokenCounter.count_tokens(p)
                    if overlap_tokens + pt > self.overlap:
                        break
                    overlap_pieces.insert(0, p)
                    overlap_tokens += pt

                current_pieces = overlap_pieces
                current_tokens = overlap_tokens

            current_pieces.append(piece)
            current_tokens += piece_tokens

        if current_pieces:
            chunks.append(join_char.join(current_pieces))

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Simple sentence splitter."""
        import re
        sentences = re.split(r'(?<=[.!?。])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
