"""Deterministic handling of explicit Requirement identifiers.

These helpers deliberately cover only strong lexical identifiers such as
``C_i``, ``MLE-QV`` and ``MPIWcapt``.  They are not a semantic entailment
judge and must not be used to infer Requirement completion.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)*(?![A-Za-z0-9])"
)


def explicit_requirement_terms(description: str) -> tuple[str, ...]:
    """Return distinctive identifiers written explicitly in a Requirement."""
    return tuple(dict.fromkeys(
        token
        for token in _IDENTIFIER.findall(description)
        if "_" in token
        or (
            sum(character.isupper() for character in token) >= 2
            and ("-" in token or any(character.islower() for character in token))
        )
    ))


def term_present(term: str, text: str) -> bool:
    """Match an identifier while tolerating common math/OCR separators."""
    parts = tuple(part for part in re.split(r"[_-]+", term) if part)
    if not parts:
        return False
    separator = r"(?:[\W_])*"
    atoms = (
        tuple(character for part in parts for character in part)
        if "-" in term and all(part.isupper() for part in parts)
        else parts
    )
    pattern = (
        r"(?<![A-Za-z0-9])"
        + separator.join(re.escape(atom) for atom in atoms)
        + r"(?![A-Za-z0-9])"
    )
    return re.search(pattern, text, re.IGNORECASE) is not None


def foreign_requirement_terms(
    requirements: Iterable[tuple[str, str]],
    primary_requirement_id: str,
) -> tuple[str, ...]:
    """Return identifiers owned only by Requirements other than ``primary``."""
    by_id = {
        requirement_id: explicit_requirement_terms(description)
        for requirement_id, description in requirements
    }
    primary = {term.casefold() for term in by_id.get(primary_requirement_id, ())}
    foreign: list[str] = []
    seen: set[str] = set()
    for requirement_id, terms in by_id.items():
        if requirement_id == primary_requirement_id:
            continue
        for term in terms:
            folded = term.casefold()
            if folded not in primary and folded not in seen:
                seen.add(folded)
                foreign.append(term)
    return tuple(foreign)
