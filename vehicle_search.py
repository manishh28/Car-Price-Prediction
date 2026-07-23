import re
from collections.abc import Iterable, Mapping


def normalize_search(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def rank_matches(
    options: Iterable[str],
    query: str,
    counts: Mapping[str, int] | None = None,
) -> list[str]:
    normalized_query = normalize_search(query)
    if not normalized_query:
        return []

    result: list[tuple[int, int, str, str]] = []
    for option in options:
        normalized_option = normalize_search(option)
        words = normalized_option.split()

        if normalized_option == normalized_query:
            relevance = 0
        elif normalized_option.startswith(normalized_query):
            relevance = 1
        elif any(word.startswith(normalized_query) for word in words):
            relevance = 2
        elif normalized_query in normalized_option:
            relevance = 3
        else:
            continue

        frequency = int(counts.get(option, 0)) if counts else 0
        result.append((relevance, -frequency, normalized_option, option))

    return [item[3] for item in sorted(result)]
