from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RetrieverWeights:

    # 网格搜素才是对的
    lexical: float = 0.30
    semantic: float = 0.50
    structural: float = 0.20


@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int = 384
    stride: int = 128
    semantic_max_length: int = 512
    structural_max_length: int = 256


@dataclass(slots=True)
class ModelConfig:
    semantic_model_name: str = "microsoft/codebert-base"
    structural_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model_name: str = "en_core_web_trf"
    projection_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.07
    margin: float = 0.2
    weights: RetrieverWeights = field(default_factory=RetrieverWeights)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 16
    grad_accumulation_steps: int = 1
    semantic_lr: float = 2e-5
    structural_lr: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 12
    warmup_ratio: float = 0.1
    fp16: bool = True
    early_stopping_patience: int = 3
    seed: int = 42
    num_workers: int = 0


@dataclass(slots=True)
class SplitConfig:
    train_size: float = 0.8
    dev_size: float = 0.1
    test_size: float = 0.1
    n_splits: int = 10


@dataclass(slots=True)
class ICLConfig:
    top_k: int = 4
    order_strategy: str = "nearest_last"
    evidence_augmented: bool = False
    enable_source_aware_calibration: bool = True
    enable_feedback_calibration: bool = True
    max_case_studies: int = 20
    max_queries: int | None = None
    api_key: str | None = None
    model_name: str = "deepseek-chat"
    url: str = "https://api.deepseek.com/chat/completions"
    timeout_seconds: int = 120


@dataclass(slots=True)
class AppConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    library: str | None = None
    trained_output_dir: Path | None = None
    rq_id: str | None = None
    rq4_query_library: str | None = None
    rq4_pool_library: str | None = None
    rq_max_folds: int | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    icl: ICLConfig = field(default_factory=ICLConfig)
    device: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
