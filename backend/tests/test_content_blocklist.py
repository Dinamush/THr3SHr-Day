"""The always-on content blocklist: nothing routes, shows or migrates into it."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.content_blocklist import (
    BLOCK_SCORE_FLOOR,
    blocked_reason_for_destination,
    blocked_reason_for_path,
    blocked_reason_for_scores,
    is_blocked_tag,
    without_blocked_tags,
)
from app.hybrid_ml import merge_ml_allowlist_scores
from app.main import app
from app.media_pooling import pool_presence
from app.services import scan_images
from app.storage import execute, fetch_all, fetch_one, to_json
from app.taxonomy import choose_best_destination, get_taxonomy, load_taxonomy, reload_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "tag",
    [
        "loli", "shota", "lolicon", "shotacon", "toddlercon", "bestiality",
        "zoophilia", "oppai_loli", "onee-loli", "onee-shota", "onii-shota",
        "kono_lolicon_domome", "mesugaki", "miniboy", "lolidom", "child",
        "child_on_child", "aged_down", "teen_(young)", "teen_young", "human_on_feral",
        "rl:loli", "rating:shota", "ロリ", "獣姦",
    ],
)
def test_blocked_tags(tag: str) -> None:
    assert is_blocked_tag(tag)


@pytest.mark.parametrize(
    "tag",
    [
        "1girl", "milf", "furry", "pokemon_(creature)", "animal_ears", "dog",
        "zoom_layer", "hololive", "bazooka", "lolita_fashion", "gothic_lolita",
        "tiger_cub", "holding_baby", "younger", "",
    ],
)
def test_allowed_tags(tag: str) -> None:
    assert not is_blocked_tag(tag)


def test_destinations_refuse_blocked_terms_in_any_part() -> None:
    for name in ("loli", "Shota", "Doujins/bestiality", "gothic_lolita", "ZooTube", "child"):
        assert blocked_reason_for_destination(name), name
    for name in ("milf", "Voyeur/panties", "Pokemon", "", None):
        assert not blocked_reason_for_destination(name), name


def test_paths_refuse_blocked_folders_and_file_names() -> None:
    assert blocked_reason_for_path("/media/ZooTube/clip.mp4")
    assert blocked_reason_for_path("/media/BestialitySexTaboo/a.mp4")
    assert blocked_reason_for_path("/media/pack/loli_set/a.png")
    assert blocked_reason_for_path("/media/pack/shota 01.png")
    assert blocked_reason_for_path("C:\\media\\ロリ\\a.png")
    # Tag-only terms are too ordinary for paths.
    assert not blocked_reason_for_path("/media/kids/birthday.png")
    assert not blocked_reason_for_path("/media/zoom/a.png")


def test_scores_refuse_at_the_floor_and_drop_below_it() -> None:
    assert blocked_reason_for_scores({"1girl": 0.99, "loli": BLOCK_SCORE_FLOOR})
    assert not blocked_reason_for_scores({"1girl": 0.99, "loli": BLOCK_SCORE_FLOOR - 0.01})
    assert without_blocked_tags({"1girl": 0.99, "loli": 0.1, "child": 0.2}) == {"1girl": 0.99}


def test_taxonomy_has_no_blocked_bucket_and_rejects_one(tmp_path: Path) -> None:
    cfg = reload_taxonomy()
    for bucket in cfg.buckets:
        for name in (bucket.id, bucket.folder, *bucket.aliases):
            assert not blocked_reason_for_destination(name), name
        tags = [
            *(ev.tag for ev in bucket.evidence),
            *(ev.tag for ev in bucket.gated_evidence),
            *bucket.veto,
            *bucket.ignore,
            *bucket.gate_tags,
        ]
        assert not [t for t in tags if is_blocked_tag(t)], bucket.id

    bad = tmp_path / "taxonomy.json"
    bad.write_text(
        json.dumps(
            {
                "buckets": [
                    {
                        "id": "zoophilia",
                        "folder": "zoophilia",
                        "priority": 1,
                        "evidence": [{"tag": "dog", "weight": 1.0}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content-blocklist"):
        load_taxonomy(bad)


def test_selected_blocked_tag_never_wins() -> None:
    reload_taxonomy()
    folder, _score, secondary = choose_best_destination(
        {"loli": 0.99, "fellatio": 0.4}, {"loli", "fellatio"}
    )
    assert folder == "fellatio"
    assert secondary == []
    assert choose_best_destination({"loli": 0.99}, {"loli"}) == (None, None, [])
    assert get_taxonomy().by_alias.get("bestiality") is None


def test_vocabularies_hold_no_blocked_tag_and_stay_index_aligned() -> None:
    """tags.csv and classes.json mirror the ML-Danbooru label order.

    Refused rows are blank placeholders, so every other label keeps its
    index; a cached copy of the model's own label file must still agree with
    every non-blank row.
    """
    with (REPO_ROOT / "tags.csv").open(encoding="utf-8", newline="") as handle:
        csv_tags = [row["tag"] for row in csv.DictReader(handle)]
    classes = json.loads((REPO_ROOT / "classes.json").read_text(encoding="utf-8"))
    assert len(csv_tags) == len(classes) == 12547
    assert csv_tags == classes
    assert not [t for t in classes if t and blocked_reason_for_destination(t)]

    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--deepghs--imgutils-models"
    cached = sorted(hub.glob("snapshots/*/mldanbooru/mldanbooru_tags.csv"))
    if not cached:
        pytest.skip("ML-Danbooru label file is not cached on this machine")
    with cached[-1].open(encoding="utf-8", newline="") as handle:
        model_labels = [row["name"] for row in csv.DictReader(handle)]
    assert len(model_labels) == len(classes)
    for index, name in enumerate(classes):
        if name:
            assert model_labels[index] == name, index
        else:
            assert blocked_reason_for_destination(model_labels[index]), index


def test_wd_engine_keeps_blocked_tags_above_a_raised_threshold() -> None:
    from app.inference_engine import InferenceEngine

    tag_names = ["general", "1girl", "loli", "solo"]
    engine = InferenceEngine()
    engine._wd_labels["SwinV2_v3"] = (tag_names, [0], [1, 2, 3], [])
    pred = np.array([0.9, 0.95, 0.4, 0.3], dtype=np.float32)
    scores = engine._wd_general_scores("SwinV2_v3", pred, 0.6, False)
    assert set(scores) == {"1girl", "loli"}
    low = np.array([0.9, 0.95, 0.2, 0.3], dtype=np.float32)
    assert set(engine._wd_general_scores("SwinV2_v3", low, 0.6, False)) == {"1girl"}


def test_one_blocked_frame_survives_presence_pooling() -> None:
    pooled = pool_presence(
        [
            (0.0, {"1girl": 0.9, "loli": 0.7}),
            (2.0, {"1girl": 0.8}),
            (4.0, {"1girl": 0.85}),
        ]
    )
    assert pooled["loli"] == pytest.approx(0.7)
    assert blocked_reason_for_scores(pooled)


def test_hybrid_merge_always_takes_blocked_ml_tags() -> None:
    merged = merge_ml_allowlist_scores({"1girl": 0.9}, {"bestiality": 0.6, "smile": 0.9})
    assert merged == {"1girl": 0.9, "bestiality": 0.6}


def test_scan_skips_blocked_folders_and_names(tmp_path: Path) -> None:
    root = tmp_path / "inbox"
    for rel in ("ok/a.png", "ZooTube/b.png", "c_loli.png", "kids/d.png"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path)
    names = sorted(p.name for p in scan_images(root).image_paths)
    assert names == ["a.png", "d.png"]


def _wait_for_run(client: TestClient, run_id: int) -> dict:
    for _ in range(200):
        status = client.get(f"/api/runs/{run_id}/status").json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_classify_run_never_stores_refused_items(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "inbox"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    scores_by_name = {
        "keep.png": {"fellatio": 0.9, "1girl": 0.8, "loli": 0.2},
        "refuse.png": {"fellatio": 0.9, "loli": 0.5},
    }
    for name in scores_by_name:
        Image.new("RGB", (8, 8)).save(root / name)
    monkeypatch.setattr(
        "app.api.extract_scores", lambda path, **_k: dict(scores_by_name[Path(path).name])
    )

    with TestClient(app) as client:
        start = client.post(
            "/api/runs/start",
            json={
                "root_repo": str(root),
                "categories_root": str(cats),
                "confidence_threshold": 0.6,
                "selected_folders": ["fellatio", "loli"],
            },
        )
        assert start.status_code == 200, start.text
        run_id = start.json()["run_id"]
        final = _wait_for_run(client, run_id)
        assert final["status"] == "completed"
        assert final["processed_images"] == 2

        items = client.get(f"/api/runs/{run_id}/items").json()
        assert [Path(item["file_path"]).name for item in items] == ["keep.png"]
        item = items[0]
        assert item["primary_tag"] == "fellatio"
        assert all(top["tag"] != "loli" for top in item["global_top_tags"])
        full = client.get(f"/api/items/{item['id']}/scores").json()["full_scores"]
        assert "loli" not in full
    stored = fetch_one("SELECT full_scores_json FROM items WHERE id = ?", (item["id"],))
    assert "loli" not in stored["full_scores_json"]


def test_settings_and_tag_search_offer_no_blocked_tag(tmp_path: Path) -> None:
    with TestClient(app) as client:
        settings = client.get("/api/settings").json()
        settings.update(
            root_repo=str(tmp_path),
            categories_root=str(tmp_path / "cats"),
            selected_tags=["loli", "fellatio", "bestiality"],
            selected_tags_nsfw=["shota", "milf"],
        )
        saved = client.put("/api/settings", json=settings).json()
        assert saved["selected_tags"] == ["fellatio"]
        assert saved["selected_tags_nsfw"] == ["milf"]

        for query in ("loli", "shota", "bestial", "child"):
            items = client.get("/api/tags", params={"query": query}).json()["items"]
            assert not [t for t in items if blocked_reason_for_destination(t)], query


def _seed_run(root: Path, cats: Path) -> int:
    return execute(
        """
        INSERT INTO runs (root_repo, categories_root, confidence_threshold, status)
        VALUES (?, ?, 0.6, 'completed')
        """,
        (str(root), str(cats)),
    )


def _seed_item(run_id: int, path: Path, tag: str, cats: Path, scores: dict) -> int:
    return execute(
        """
        INSERT INTO items (
            run_id, file_path, relative_path, primary_tag, primary_score,
            secondary_json, full_scores_json, suggested_destination, final_tag,
            final_destination, status, needs_review
        ) VALUES (?, ?, ?, ?, 0.9, '[]', ?, ?, ?, ?, 'approved', 0)
        """,
        (
            run_id,
            str(path),
            path.name,
            tag,
            to_json(scores),
            str(cats / tag),
            tag,
            str(cats / tag),
        ),
    )


def test_review_and_migrate_refuse_blocked_items(tmp_path: Path) -> None:
    root = tmp_path / "inbox"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    ok = root / "ok.png"
    tagged = root / "tagged.png"
    for path in (ok, tagged):
        Image.new("RGB", (8, 8)).save(path)

    with TestClient(app) as client:
        run_id = _seed_run(root, cats)
        ok_id = _seed_item(run_id, ok, "fellatio", cats, {"fellatio": 0.9})
        # Stored before the blocklist: routed elsewhere, but its scores refuse it.
        tagged_id = _seed_item(run_id, tagged, "fellatio", cats, {"fellatio": 0.9, "shota": 0.8})

        refused = client.patch(f"/api/items/{ok_id}", json={"final_tag": "loli"})
        assert refused.status_code == 400
        refused = client.patch(f"/api/items/{tagged_id}", json={"status": "approved"})
        assert refused.status_code == 400
        batch = client.post(
            f"/api/runs/{run_id}/batch",
            json={"item_ids": [ok_id], "status": "approved", "final_tag": "bestiality"},
        ).json()
        assert batch == {"updated": 0, "skipped": 1}

        body = client.post(
            f"/api/runs/{run_id}/migrate", json={"mode": "copy", "create_missing_folders": True}
        ).json()
    assert body["migrated_count"] == 1
    assert [r["item_id"] for r in body["results"]] == [tagged_id]
    assert (cats / "fellatio" / "ok.png").is_file()
    assert not (cats / "fellatio" / "tagged.png").exists()
    assert tagged.is_file()


def test_startup_purges_rows_routed_into_blocked_folders(tmp_path: Path) -> None:
    root = tmp_path / "inbox"
    cats = tmp_path / "cats"
    run_id = _seed_run(root, cats)
    kept = _seed_item(run_id, root / "a.png", "fellatio", cats, {})
    purged = _seed_item(run_id, root / "b.png", "loli", cats, {})
    execute(
        "INSERT INTO item_global_tops (item_id, run_id, tops_json) VALUES (?, ?, '[]')",
        (purged, run_id),
    )

    with TestClient(app):
        pass

    assert [row["id"] for row in fetch_all("SELECT id FROM items")] == [kept]
    assert fetch_all("SELECT item_id FROM item_global_tops") == []
