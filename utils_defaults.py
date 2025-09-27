from typing import Optional

def default_n_topics(n_docs: Optional[int]) -> int:
    """Heuristique simple pour le nombre par défaut de topics.
    - On prend environ n_docs // 5
    - Borne entre 2 et 20 pour éviter extrêmes
    """
    if not n_docs or n_docs <= 0:
        return 6
    return max(2, min(20, max(2, n_docs // 5)))

def default_ber_min_topic_size(n_docs: Optional[int]) -> int:
    """Heuristique par défaut pour min_topic_size de BERTopic.
    - Environ n_docs // 15, borné entre 2 et 10.
    """
    if not n_docs or n_docs <= 0:
        return 2
    return max(2, min(10, max(2, n_docs // 15)))
