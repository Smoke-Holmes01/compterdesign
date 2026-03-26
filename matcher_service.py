import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional

from match_groups_to_objs import extract_primary_contour
from match_groups_to_objs import load_binary
from match_groups_to_objs import load_model_views
from match_groups_to_objs import score_one_image_against_one_model
from preprocess_groups import preprocess_one_image


PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_saved_path(project_root: Path, raw_path: str) -> str:
    saved_path = Path(raw_path)
    if saved_path.is_absolute():
        return str(saved_path)
    return str((project_root / saved_path).resolve())


def _load_render_meta(project_root: Path, render_root: Path) -> dict[str, list[dict[str, Any]]]:
    models: dict[str, list[dict[str, Any]]] = {}
    if not render_root.exists():
        raise FileNotFoundError(f"Render root not found: {render_root}")

    for model_dir in sorted(render_root.iterdir(), key=lambda item: item.name):
        if not model_dir.is_dir():
            continue

        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            continue

        with meta_path.open("r", encoding="utf-8") as file:
            views = json.load(file)

        normalized_views: list[dict[str, Any]] = []
        for view in views:
            normalized_view = dict(view)
            normalized_view["mask_path"] = _resolve_saved_path(project_root, view["mask_path"])
            normalized_view["edge_path"] = _resolve_saved_path(project_root, view["edge_path"])
            normalized_views.append(normalized_view)

        models[model_dir.name] = normalized_views

    if not models:
        raise ValueError(f"No rendered model metadata found under: {render_root}")

    return models


def _load_preprocessed_image(record: dict[str, Any]) -> dict[str, Any]:
    mask = load_binary(record["mask_path"])
    edge = load_binary(record["edge_path"])
    return {
        "name": record["name"],
        "mask": mask,
        "edge": edge,
        "contour": extract_primary_contour(mask),
    }


@dataclass
class SingleImageMatcherConfig:
    project_root: Path = PROJECT_ROOT
    render_root: Path = PROJECT_ROOT / "outputs" / "renders"
    query_root: Path = PROJECT_ROOT / "outputs" / "agent_queries"
    out_size: int = 512


class SingleImageMatcher:
    def __init__(self, config: Optional[SingleImageMatcherConfig] = None):
        self.config = config or SingleImageMatcherConfig()
        self._render_meta_cache: Optional[dict[str, list[dict[str, Any]]]] = None

    def _get_render_meta(self) -> dict[str, list[dict[str, Any]]]:
        if self._render_meta_cache is None:
            self._render_meta_cache = _load_render_meta(
                project_root=self.config.project_root,
                render_root=self.config.render_root,
            )
        return self._render_meta_cache

    def _prepare_query_assets(self, image_path: str, query_id: str) -> tuple[dict[str, Any], Path]:
        query_dir = self.config.query_root / query_id
        preprocessed_dir = query_dir / "preprocessed"
        _ensure_dir(preprocessed_dir)

        record = preprocess_one_image(
            image_path=image_path,
            out_dir=str(preprocessed_dir),
            out_size=self.config.out_size,
        )
        loaded_record = _load_preprocessed_image(record)
        return loaded_record, query_dir

    def match_image(self, image_path: str, top_k: int = 3, query_id: Optional[str] = None) -> dict[str, Any]:
        normalized_top_k = max(1, int(top_k))
        normalized_query_id = query_id or Path(image_path).stem

        image_record, query_dir = self._prepare_query_assets(
            image_path=image_path,
            query_id=normalized_query_id,
        )

        scored_models: list[dict[str, Any]] = []
        render_meta = self._get_render_meta()

        for model_name, model_views_meta in render_meta.items():
            model_views = load_model_views(model_views_meta)
            best_view = score_one_image_against_one_model(image_record, model_views)
            scored_models.append(
                {
                    "model_name": model_name,
                    "score": float(best_view["total_score"]),
                    "best_view": best_view,
                }
            )

        scored_models.sort(key=lambda item: item["score"], reverse=True)

        result = {
            "query_id": normalized_query_id,
            "image_name": image_record["name"],
            "top_k": normalized_top_k,
            "total_models": len(scored_models),
            "best_match": scored_models[0] if scored_models else None,
            "matches": scored_models[:normalized_top_k],
            "query_assets": {
                "query_dir": str(query_dir.resolve()),
                "mask_path": str((query_dir / "preprocessed" / f"{image_record['name']}_mask.png").resolve()),
                "edge_path": str((query_dir / "preprocessed" / f"{image_record['name']}_edge.png").resolve()),
                "rgb_path": str((query_dir / "preprocessed" / f"{image_record['name']}_rgb.png").resolve()),
            },
        }

        result_path = query_dir / "match_result.json"
        _ensure_dir(query_dir)
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        result["result_path"] = str(result_path.resolve())
        return result


def match_single_image(image_path: str, top_k: int = 3, query_id: Optional[str] = None) -> dict[str, Any]:
    matcher = SingleImageMatcher()
    return matcher.match_image(image_path=image_path, top_k=top_k, query_id=query_id)
