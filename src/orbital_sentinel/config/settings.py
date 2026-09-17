from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw/esa_kelvins"
    train_file: str = "train_data.csv"
    test_file: str = "test_data.csv"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"


@dataclass
class TargetConfig:
    column: str = "risk"
    description: str = "log10 collision probability"
    floor_value: float = -30.0


@dataclass
class FeaturesConfig:
    id_columns: List[str] = field(default_factory=lambda: ["event_id", "mission_id"])
    target_columns: List[str] = field(default_factory=lambda: ["risk"])
    leakage_columns: List[str] = field(
        default_factory=lambda: ["max_risk_estimate", "max_risk_scaling"]
    )
    categorical_columns: List[str] = field(default_factory=lambda: ["c_object_type"])
    space_weather_columns: List[str] = field(
        default_factory=lambda: ["F10", "F3M", "SSN", "AP"]
    )


@dataclass
class SplittingConfig:
    test_size: float = 0.2
    random_state: int = 42
    strategy: str = "event_based"


@dataclass
class Settings:
    data: DataConfig = field(default_factory=DataConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    project_root: Path = field(default_factory=lambda: Path.cwd())


_settings: Optional[Settings] = None


def _find_project_root() -> Path:
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "config.yaml").exists():
            return parent
    return current


def _load_from_yaml(path: Path) -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project_root = path.parent.parent

    data_cfg = DataConfig(**raw.get("data", {}))
    target_cfg = TargetConfig(**raw.get("target", {}))
    features_cfg = FeaturesConfig(**raw.get("features", {}))
    splitting_cfg = SplittingConfig(**raw.get("splitting", {}))

    return Settings(
        data=data_cfg,
        target=target_cfg,
        features=features_cfg,
        splitting=splitting_cfg,
        project_root=project_root,
    )


def get_settings(config_path: Optional[Path] = None) -> Settings:
    global _settings
    if _settings is not None and config_path is None:
        return _settings

    if config_path is None:
        project_root = _find_project_root()
        config_path = project_root / "configs" / "config.yaml"

    if config_path.exists():
        _settings = _load_from_yaml(config_path)
    else:
        _settings = Settings()

    return _settings
