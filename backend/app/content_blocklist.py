"""Content THR3SHR never routes, suggests, exports or migrates.

It refuses sexual content involving minors, drawn included, and bestiality.
It is always on: there is no setting, folder or option that turns it off.

The checks share one set of term lists:

* :func:`blocked_reason_for_scores` refuses an item whose tagger scores carry
  a blocked tag at or above :data:`BLOCK_SCORE_FLOOR`.  A refused item is
  skipped: it never becomes a reviewable row, so it cannot be approved or
  migrated, and nothing about it is stored.
* :func:`without_blocked_tags` drops blocked tags from a score map before it
  is stored or shown, so a tag below the floor never surfaces either.
* :func:`blocked_reason_for_destination` refuses a destination folder or a
  selected tag that names a blocked term.
* :func:`blocked_reason_for_path` refuses a file whose folder or file name
  names a blocked term or a blocked source.

A tag-based filter cannot see what an untagged image shows; it removes what
the taggers label.  Matching errs towards blocking: a minor-coded tag blocks
the item whatever else it shows.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import lru_cache

#: Blocked anywhere they appear as a whole word: a tag, a folder name, a file
#: name or a selected destination.
STRONG_TERMS: frozenset[str] = frozenset({
    # minors
    "loli", "lolicon", "lolita", "shota", "shotacon", "toddlercon",
    "underage", "preteen", "prepubescent", "pedo", "pedophile", "pedophilia",
    "paedophile", "paedophilia", "jailbait", "child_porn",
    # bestiality
    "bestiality", "zoophilia", "zoophile", "zoosex", "zoo_sex", "animal_sex",
    "human_on_feral", "feral_on_human",
})

#: Blocked only as a whole *tag* (or a whole destination name): in a file or
#: folder name they are too ordinary ("kid", "young"), but as a tagger tag
#: they mark a minor or an animal partner.
TAG_ONLY_TERMS: frozenset[str] = frozenset({
    "child", "children", "kid", "kids", "young", "cub", "toddler", "infant",
    "baby", "minor", "aged_down", "young_girl", "young_boy", "little_girl",
    "little_boy", "child_on_child", "teen_(young)", "elementary_school_student",
    "middle_school_student", "k9", "dog_on_human", "human_on_animal",
    "animal_on_human", "feral_penetrated", "feral_penetrating_human",
    # Minor-coded tags in the WD and ML-Danbooru vocabularies that no word
    # rule catches.
    "mesugaki", "miniboy", "lolidom",
})

#: Substrings for scripts without word breaks (Japanese tags and file names).
SUBSTRING_TERMS: tuple[str, ...] = (
    "ロリ", "ショタ", "幼女", "幼児", "児童", "小学生", "園児", "獣姦",
)

#: Strong terms that do not block when they are only one word of a longer
#: tag: the lolita_fashion and gothic_lolita tags name a clothing style.  As
#: a whole tag, a folder name or a file name they still block.
TAG_WORD_EXEMPT: frozenset[str] = frozenset({"lolita"})

#: Download sources whose whole catalogue is bestiality.  A folder named
#: after one of them (a downloader's per-source folder) blocks every file in
#: it.
BLOCKED_SOURCE_FOLDERS: frozenset[str] = frozenset({"zootube", "bestialitysextaboo", "bst"})

#: Score at which a blocked tag refuses the item.  It matches the default WD
#: general-tag cutoff and sits well below the default assignment threshold,
#: so a tag weak enough to stay off the review screen still refuses the item.
BLOCK_SCORE_FLOOR = 0.35

REFUSED_REASON = "Refused by the content blocklist."

_WORD_RE = re.compile(r"[^\W_]+(?:_[^\W_]+)*", re.UNICODE)


def _key(value: str) -> str:
    """Lower case with every run of non-alphanumerics folded to one ``_``.

    Tagger keys drop brackets (``teen_(young)`` scores as ``teen_young``), so
    the term lists and every tag pass through the same folding.
    """
    return "_".join(_WORD_RE.findall(str(value or "").lower().replace("_", " ")))


_STRONG_KEYS = frozenset(_key(t) for t in STRONG_TERMS)
_TAG_ONLY_KEYS = frozenset(_key(t) for t in TAG_ONLY_TERMS)
# Strong terms matched as a run of whole words inside a longer tag.
_TAG_RUNS = tuple(f"_{t}_" for t in sorted(_STRONG_KEYS - TAG_WORD_EXEMPT))


def _words(text: str) -> set[str]:
    """Whole words and underscore-joined runs ("human_on_feral")."""
    lowered = str(text or "").lower()
    found = set(_WORD_RE.findall(lowered))
    for run in list(found):
        if "_" in run:
            found.update(run.split("_"))
    return found


def _substring_hit(text: str) -> str:
    for term in SUBSTRING_TERMS:
        if term in text:
            return term
    return ""


def is_blocked_tag(tag: str) -> bool:
    """True for a tagger tag (or a selected tag) that marks blocked content.

    A namespace prefix (``rl:``, ``rating:``) is ignored.  The tag blocks as a
    whole term, or when one of its words is a strong term (``oppai_loli``).
    """
    return _is_blocked_tag(str(tag or ""))


# Tagger vocabularies are a few thousand names, checked once per image.
@lru_cache(maxsize=65536)
def _is_blocked_tag(raw: str) -> bool:
    bare = raw.split(":", 1)[1] if ":" in raw else raw
    key = _key(bare)
    if not key:
        return False
    if key in _STRONG_KEYS or key in _TAG_ONLY_KEYS:
        return True
    padded = f"_{key}_"
    if any(run in padded for run in _TAG_RUNS):
        return True
    return bool(_substring_hit(bare))


def blocked_reason_for_text(text: str) -> str:
    """Reason when free text (a folder or file name) names a strong term."""
    words = {_key(w) for w in _words(text)}
    hit = words & _STRONG_KEYS
    if hit:
        return f"names blocked term {sorted(hit)[0]!r}"
    sub = _substring_hit(str(text or ""))
    return f"names blocked term {sub!r}" if sub else ""


def blocked_tags_in_scores(
    scores: Mapping[str, float], floor: float = BLOCK_SCORE_FLOOR
) -> list[str]:
    """Blocked tags scored at or above *floor*, highest first."""
    hits: list[tuple[float, str]] = []
    for tag, score in (scores or {}).items():
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if value >= floor and is_blocked_tag(tag):
            hits.append((value, str(tag)))
    hits.sort(key=lambda item: (-item[0], item[1]))
    return [tag for _value, tag in hits]


def blocked_reason_for_scores(
    scores: Mapping[str, float], floor: float = BLOCK_SCORE_FLOOR
) -> str:
    """Reason when an item's tagger scores refuse it, else ``""``."""
    return REFUSED_REASON if blocked_tags_in_scores(scores, floor) else ""


def without_blocked_tags(scores: Mapping[str, float]) -> dict[str, float]:
    """Copy of *scores* with every blocked tag dropped, whatever its score."""
    return {tag: value for tag, value in (scores or {}).items() if not is_blocked_tag(tag)}


def _name_parts(name: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", str(name or "")) if part.strip()]


def blocked_reason_for_destination(name: str | None) -> str:
    """Reason when a destination folder or selected tag is refused.

    *name* may be a nested folder (``Voyeur/panties``) or a full destination
    path; every part of it is checked.
    """
    for part in _name_parts(name or ""):
        if part.strip().casefold() in BLOCKED_SOURCE_FOLDERS:
            return REFUSED_REASON
        if is_blocked_tag(part) or blocked_reason_for_text(part):
            return REFUSED_REASON
    return ""


def is_blocked_destination(name: str | None) -> bool:
    return bool(blocked_reason_for_destination(name))


def without_blocked_destinations(names: Iterable[str]) -> list[str]:
    """*names* in order, without the ones the blocklist refuses."""
    return [name for name in names if not is_blocked_destination(name)]


def blocked_reason_for_path(path: str | None) -> str:
    """Reason when a file is off limits: a folder on its path is named after a
    blocked source or names a blocked term, or its file name does."""
    for part in _name_parts(path or ""):
        if part.strip().casefold() in BLOCKED_SOURCE_FOLDERS:
            return REFUSED_REASON
        if blocked_reason_for_text(part):
            return REFUSED_REASON
    return ""
