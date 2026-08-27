"""为普通检索与参考文献检索提供统一的确定性英文词法提取。"""

from __future__ import annotations

from collections.abc import Sequence
import re


# 问句结构词和检索元话语没有区分度，不能占用最多五个关键词名额。
LEXICAL_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "author", "authors", "be",
        "been", "between", "by", "compare", "concern", "concerns", "context", "did",
        "differ", "do", "does", "during", "for", "from", "full", "how",
        "in", "into", "is", "it", "its", "journal", "journals", "like",
        "literature", "method", "model", "of", "on", "or", "original",
        "paper", "papers", "publication", "publications", "reference",
        "referenced", "references", "respectively", "result", "results",
        "study", "studies", "that", "the", "their", "them", "these", "they",
        "this", "titled", "titles", "to", "traditional", "two", "used",
        "using", "variant", "variants", "was", "were", "what", "when",
        "where", "which", "who", "why", "with", "work", "works", "year",
        "years",
    }
)

# 支持 PM2.5、LSTM-EMVE、D²STGNN†、shapeDTW 等不能按普通单词切分的科研实体。
_TOKEN_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[.\-:²³][A-Za-z0-9]+)*(?:[†‡])?"
    r"|\d+(?:\.\d+)?%"
    r"|\d{2,4}"
)
_HOUR_PATTERN = re.compile(r"(?<!\d)(\d{1,4})\s*(?:小时|小時|hours?|hrs?)(?![A-Za-z])", re.I)
_PARENTHETICAL_PHRASE_PATTERN = re.compile(r"[（(]\s*([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){1,3})\s*[）)]")
_MODIFIER_ENTITY_PATTERN = re.compile(
    r"\b([a-z][a-z-]{2,})\s+([A-Za-z][A-Za-z0-9]*(?:[.\-:²³][A-Za-z0-9]+)*(?:[†‡])?)\b"
)
_CAPITALIZED_PHRASE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9]*(?:[.\-:²³][A-Za-z0-9]+)*(?:[†‡])?"
    r"(?:\s+[A-Z][A-Za-z0-9]*(?:[.\-:²³][A-Za-z0-9]+)*(?:[†‡])?){1,3})"
    r"(?![A-Za-z0-9])"
)


def _term_key(value: str) -> str:
    """生成大小写不敏感且空白稳定的去重键。"""
    return " ".join(value.split()).casefold()


def _is_symbolic_entity(value: str) -> bool:
    """判断词是否具有缩写、混合大小写、数字或科研符号等高区分度特征。"""
    if any(character.isdigit() or character in ".-:²³†‡%" for character in value):
        return True
    letters = [character for character in value if character.isalpha()]
    if not letters:
        return False
    upper_count = sum(character.isupper() for character in letters)
    # 全大写缩写和 shapeDTW 这类内部大写名称均视为科研实体；普通句首单词不会通过。
    return upper_count >= 2 or (upper_count >= 1 and any(character.islower() for character in letters[1:]) and any(character.isupper() for character in letters[1:]))


def _phrase_is_useful(value: str) -> bool:
    """过滤全由停用词组成或以问句结构词开头的低区分度短语。"""
    words = value.split()
    if len(words) < 2 or words[0].casefold() in LEXICAL_STOPWORDS:
        return False
    meaningful = [word for word in words if word.casefold() not in LEXICAL_STOPWORDS]
    return len(meaningful) >= 2


def _add_candidate(
    candidates: list[tuple[int, int, str]],
    seen: set[str],
    *,
    value: str,
    score: int,
    position: int,
) -> None:
    """规范化并加入候选；同一词只保留第一次出现的最高质量版本。"""
    normalized = " ".join(value.strip(" ,.;:!?？。；，").split())
    key = _term_key(normalized)
    if not normalized or key in seen or key in LEXICAL_STOPWORDS:
        return
    seen.add(key)
    candidates.append((score, position, normalized))


def extract_lexical_terms(query: str, max_terms: int = 5) -> tuple[str, ...]:
    """从原问题提取最多五个可追溯的高区分度英文实体、数字或专业短语。

    函数不翻译、不扩写缩写，也不依赖论文领域词典。排序优先保护科研符号实体，
    然后保留年份、数值约束和多词专业短语；同分时保持原问题中的出现顺序。
    """
    if max_terms < 1:
        return ()
    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    capitalized_phrase_spans: list[tuple[int, int]] = []

    for match in _CAPITALIZED_PHRASE_PATTERN.finditer(query):
        phrase = match.group(1)
        if phrase.split()[0].casefold() not in LEXICAL_STOPWORDS and any(
            word.casefold() not in LEXICAL_STOPWORDS
            for word in phrase.split()
        ):
            capitalized_phrase_spans.append(match.span(1))
            _add_candidate(
                candidates,
                seen,
                value=phrase,
                score=140 + len(phrase),
                position=match.start(1),
            )

    hour_spans: list[tuple[int, int]] = []
    for match in _HOUR_PATTERN.finditer(query):
        hour_spans.append(match.span())
        _add_candidate(
            candidates,
            seen,
            value=f"{match.group(1)}-hour",
            score=125,
            position=match.start(),
        )

    for match in _TOKEN_PATTERN.finditer(query):
        value = match.group(0)
        key = value.casefold()
        if key in LEXICAL_STOPWORDS:
            continue
        if value.isdigit() and any(start <= match.start() < end for start, end in hour_spans):
            # “72 小时”已经规范成 72-hour，不再让裸数字重复占一个名额。
            continue
        if any(start <= match.start() and match.end() <= end for start, end in capitalized_phrase_spans):
            # Estimation Gate、Graph WaveNet 等实体已作为整体加入，不再保留拆开的组成词。
            continue
        if _is_symbolic_entity(value):
            # 更长的模型名优先于过短缩写，避免主方法被 ED 等短词挤出。
            score = 120 + min(len(value), 20)
        elif value.isdigit() or value.endswith("%"):
            score = 105
        elif value[0].isupper() and value[1:].islower():
            # Itakura 等英文专名对 References 检索有价值，但优先级低于模型和数字约束。
            score = 110
        else:
            continue
        _add_candidate(candidates, seen, value=value, score=score, position=match.start())

    for match in _PARENTHETICAL_PHRASE_PATTERN.finditer(query):
        phrase = match.group(1)
        if _phrase_is_useful(phrase):
            _add_candidate(
                candidates,
                seen,
                value=phrase,
                score=100 + len(phrase.split()),
                position=match.start(1),
            )

    # 按停用词切开连续英文内容词，再组成 2～4 词短语，避免得到“speech recognition paper and”。
    phrase_run: list[tuple[str, int]] = []

    def flush_phrase_run() -> None:
        """把当前连续内容词收束为一个专业短语候选。"""
        if len(phrase_run) >= 2:
            words = phrase_run[-4:]
            _add_candidate(
                candidates,
                seen,
                value=" ".join(word for word, _ in words),
                score=95 + len(words),
                position=words[0][1],
            )
        phrase_run.clear()

    for match in re.finditer(r"[A-Za-z][A-Za-z-]*", query):
        word = match.group(0)
        if word.casefold() in LEXICAL_STOPWORDS or _is_symbolic_entity(word) or (
            word[0].isupper() and match.start() > 0
        ):
            flush_phrase_run()
            continue
        phrase_run.append((word, match.start()))
    flush_phrase_run()

    for match in _MODIFIER_ENTITY_PATTERN.finditer(query):
        phrase = match.group(0)
        if match.group(1).casefold() not in LEXICAL_STOPWORDS and _is_symbolic_entity(match.group(2)):
            _add_candidate(
                candidates,
                seen,
                value=phrase,
                score=118,
                position=match.start(),
            )

    candidates.sort(key=lambda item: (-item[0], item[1], item[2].casefold()))
    return tuple(value for _, _, value in candidates[:max_terms])


def normalize_lexical_terms(values: Sequence[str], max_terms: int = 5) -> tuple[str, ...]:
    """清理外部关键词并过滤明确问句停用词，保持原有先后顺序。"""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).strip(" ,.;:!?？。；，").split())
        key = _term_key(cleaned)
        if not cleaned or key in seen or key in LEXICAL_STOPWORDS:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) == max_terms:
            break
    return tuple(normalized)


def merge_lexical_terms(
    llm_terms: Sequence[str],
    deterministic_terms: Sequence[str],
    max_terms: int = 5,
) -> tuple[str, ...]:
    """合并 LLM 与确定性关键词，并优先保护原问题中可直接观察到的实体。"""
    # 原问题实体先占位，避免 LLM 泛化短语把模型名、数值或变体标记挤出前五。
    return normalize_lexical_terms(
        (*deterministic_terms, *llm_terms),
        max_terms=max_terms,
    )
