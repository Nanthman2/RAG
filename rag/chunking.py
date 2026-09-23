"""Découpage de texte en chunks avec chevauchement, en respectant les paragraphes."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(para) <= chunk_size:
            current = para
        else:
            # Paragraphe trop long : découpage brut avec chevauchement.
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = end - overlap
            current = ""

    if current:
        chunks.append(current)

    # Chevauchement entre chunks consécutifs pour préserver le contexte aux frontières.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, curr in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            overlapped.append(f"{tail}\n\n{curr}")
        return overlapped

    return chunks
