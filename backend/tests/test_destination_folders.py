"""Selected folders are what a run competes for."""

from __future__ import annotations

from app.services import destination_folders_for_tagger


def test_selected_folders_are_kept_for_each_tagger() -> None:
    selected = ["Voyeur", "fellatio", "monster_girl"]
    assert destination_folders_for_tagger(selected, "ml_danbooru") == selected
    assert destination_folders_for_tagger(selected, "wd_swinv2_v3") == selected
    assert destination_folders_for_tagger(
        ["real_life"], "ml_danbooru", real_life_filter=True
    ) == ["real_life"]
