from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from statistics import mean

from .bm25 import BM25Retriever
from .clues import (
    export_augmented_samples,
    infer_sample_decision_profile,
    infer_feedback_rule_hits,
    render_feedback_rule_text,
    render_demo_reason,
    render_sample_clue_text,
)
from .config import AppConfig
from .data import ApiSample, build_kfold_splits, group_samples_by_library, load_samples, stratified_split
from .error_analysis import build_error_analysis
from .evaluation import case_studies, classification_metrics, retrieval_metrics
from .feedback import apply_feedback_calibration
from .icl import (
    DeepSeekClient,
    build_prompt,
    calibrate_prediction,
    parse_prediction,
    select_demonstrations,
)
from .reason_feedback import build_preprocessing_feedback, export_preprocessing_feedback_csv
from .retrieval import HybridRetriever
from .trainer import RetrieverTrainer, TrainingArtifacts
from .utils import dump_json, ensure_dir


def _subset_config(config: AppConfig, name: str) -> AppConfig:
    subset = deepcopy(config)
    subset.output_dir = config.output_dir / name
    return subset


def _metric_average(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted(
        {
            key
            for row in metric_rows
            for key, value in row.items()
            if isinstance(value, (int, float))
        }
    )
    return {key: mean([float(row[key]) for row in metric_rows if key in row]) for key in keys}


def _safe_classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    if not labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return classification_metrics(labels, predictions)


def _source_metrics(
    queries: list[ApiSample],
    predictions: list[int],
) -> dict[str, dict[str, float]]:
    grouped_labels: dict[str, list[int]] = {}
    grouped_predictions: dict[str, list[int]] = {}
    for query, prediction in zip(queries, predictions, strict=True):
        grouped_labels.setdefault(query.source, []).append(query.label)
        grouped_predictions.setdefault(query.source, []).append(prediction)
    return {
        source: _safe_classification_metrics(grouped_labels[source], grouped_predictions[source])
        for source in sorted(grouped_labels)
    }


def _confusion_counts_by_source(
    queries: list[ApiSample],
    predictions: list[int],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for query, prediction in zip(queries, predictions, strict=True):
        current = counts.setdefault(query.source, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
        if query.label == 1 and prediction == 1:
            current["tp"] += 1
        elif query.label == 0 and prediction == 0:
            current["tn"] += 1
        elif query.label == 0 and prediction == 1:
            current["fp"] += 1
        else:
            current["fn"] += 1
    return counts


def _summarize_query(query: ApiSample, reason: str) -> dict[str, str | int]:
    return {
        "sample_id": query.sample_id,
        "api": query.api,
        "dataset": query.dataset,
        "source": query.source,
        "label": query.label,
        "reason": reason,
        "fragment_preview": query.fragment[:500],
    }


def _bucket_so_errors(prediction_rows: list[dict], queries: list[ApiSample], prompt_rows: list[list[dict]]) -> dict[str, dict[str, list[dict]]]:
    by_id = {query.sample_id: query for query in queries}
    false_positives = {
        "low_quality_qa_mistaken_as_positive": [],
        "useful_answer_but_wrong_api_focus": [],
    }
    false_negatives = {
        "short_but_valid_api_guidance_missed": [],
        "other_missed_positive": [],
    }

    for row, prompt_demo_rows in zip(prediction_rows, prompt_rows, strict=True):
        query = by_id[row["sample_id"]]
        if query.source != "SO":
            continue
        if row["gold_label"] == 0 and row["predicted_label"] == 1:
            wrong_api_focus = any(
                demo["label"] == 0 and demo["api"] != query.api for demo in prompt_demo_rows
            )
            bucket = (
                "useful_answer_but_wrong_api_focus"
                if wrong_api_focus
                else "low_quality_qa_mistaken_as_positive"
            )
            false_positives[bucket].append(_summarize_query(query, row["reason"]))
        elif row["gold_label"] == 1 and row["predicted_label"] == 0:
            bucket = (
                "short_but_valid_api_guidance_missed"
                if len(query.fragment) <= 900
                else "other_missed_positive"
            )
            false_negatives[bucket].append(_summarize_query(query, row["reason"]))

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _resolve_checkpoint_path(config: AppConfig) -> Path:
    if not config.trained_output_dir:
        raise ValueError("RQ experiment requires a trained_output_dir that contains checkpoints/best.pt")
    checkpoint_path = Path(config.trained_output_dir) / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _load_trained_artifacts(config: AppConfig) -> tuple[list[ApiSample], dict[str, list[ApiSample]], TrainingArtifacts]:
    samples = load_samples(config.data_dir)
    grouped = group_samples_by_library(samples)
    trainer = RetrieverTrainer(config)
    artifacts = trainer.load_from_checkpoint(_resolve_checkpoint_path(config), samples)
    return samples, grouped, artifacts


def _validate_library(config: AppConfig, grouped: dict[str, list[ApiSample]]) -> None:
    if config.library and config.library not in grouped:
        raise ValueError(f"Unknown library: {config.library}. Available: {', '.join(grouped)}")


def train_retriever(config: AppConfig) -> dict:
    samples = load_samples(config.data_dir)
    train_samples, dev_samples, test_samples = stratified_split(
        samples,
        train_size=config.split.train_size,
        dev_size=config.split.dev_size,
        test_size=config.split.test_size,
        seed=config.training.seed,
    )
    ensure_dir(config.output_dir)
    trainer = RetrieverTrainer(config)
    artifacts = trainer.fit(train_samples, dev_samples, extra_samples=samples)
    test_metrics = trainer.evaluate(
        train_samples,
        test_samples,
        artifacts.sop_strings,
        artifacts.semantic_encoder,
        artifacts.structural_encoder,
        artifacts.bm25,
    )
    summary = {
        "mode": "global_train_all_9_pairs",
        "num_samples": len(samples),
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "test_samples": len(test_samples),
        "test_metrics": test_metrics,
    }
    dump_json(test_metrics, config.output_dir / "test_retrieval_metrics.json")
    dump_json(summary, config.output_dir / "train_summary.json")
    return {
        "samples": samples,
        "train_samples": train_samples,
        "dev_samples": dev_samples,
        "test_samples": test_samples,
        "artifacts": artifacts,
        "test_metrics": test_metrics,
        "summary": summary,
    }


def run_icl_fold(
    config: AppConfig,
    pool: list[ApiSample],
    queries: list[ApiSample],
    artifacts: TrainingArtifacts,
    output_name: str,
) -> dict:
    trainer = RetrieverTrainer(config)
    ensure_dir(config.output_dir)
    bm25 = BM25Retriever()
    bm25.fit(pool)
    if config.icl.max_queries is not None:
        queries = queries[: config.icl.max_queries]
    pool_semantic, pool_structural = trainer._encode_pool(
        pool,
        artifacts.sop_strings,
        artifacts.semantic_encoder,
        artifacts.structural_encoder,
    )
    retriever = HybridRetriever(bm25, config.model.weights)
    client = DeepSeekClient(config.icl)

    labels = []
    predictions = []
    rankings = []
    prediction_rows = []
    prompt_rows_by_query: list[list[dict[str, str | int | float]]] = []
    api_prior_stats: dict[tuple[str, str], dict[str, float | int]] = {}
    for sample in pool:
        key = (sample.source, sample.api)
        if key not in api_prior_stats:
            api_prior_stats[key] = {"support": 0, "positive": 0}
        api_prior_stats[key]["support"] = int(api_prior_stats[key]["support"]) + 1
        api_prior_stats[key]["positive"] = int(api_prior_stats[key]["positive"]) + int(sample.label)
    for query in queries:
        rows = retriever.retrieve(
            query=query,
            pool=pool,
            pool_semantic=pool_semantic,
            pool_structural=pool_structural,
            semantic_encoder=artifacts.semantic_encoder,
            structural_encoder=artifacts.structural_encoder,
            sop_string=artifacts.sop_strings[query.sample_id],
            top_k=max(24, config.icl.top_k * 6, 10),
            device=trainer.device,
        )
        demonstrations = select_demonstrations(query, rows, config.icl)
        prompt = build_prompt(query, demonstrations, config.icl)
        result = calibrate_prediction(parse_prediction(client.chat(prompt)), query, demonstrations, config.icl)
        prompt_demo_rows = [
            {
                "sample_id": demo.candidate.sample_id,
                "api": demo.candidate.api,
                "label": demo.candidate.label,
                "source": demo.candidate.source,
                "fused_score": demo.fused_score,
                "clue_features": render_sample_clue_text(demo.candidate),
                "decision_profile": infer_sample_decision_profile(demo.candidate),
                "feedback_rule_hits": infer_feedback_rule_hits(demo.candidate),
                "feedback_rule_text": render_feedback_rule_text(demo.candidate),
                "heuristic_rationale": render_demo_reason(demo.candidate),
            }
            for demo in demonstrations
        ]
        prediction_row = {
            "sample_id": query.sample_id,
            "api": query.api,
            "source": query.source,
            "dataset": query.dataset,
            "gold_label": query.label,
            "predicted_label": result.label,
            "llm_predicted_label": result.llm_label,
            "calibration_applied": result.calibration_applied,
            "calibration_note": result.calibration_note,
            "reason": result.reason,
            "raw_response": result.raw_response,
            "query_clue_features": render_sample_clue_text(query),
            "query_decision_profile": infer_sample_decision_profile(query),
            "query_feedback_rule_hits": infer_feedback_rule_hits(query),
            "query_feedback_rule_text": render_feedback_rule_text(query),
            "prompt_demo_labels": [demo.candidate.label for demo in demonstrations],
            "prompt_demo_apis": [demo.candidate.api for demo in demonstrations],
            "prompt_demo_sources": [demo.candidate.source for demo in demonstrations],
            "prompt_demo_decision_profiles": [
                infer_sample_decision_profile(demo.candidate) for demo in demonstrations
            ],
            "prompt_demo_feedback_rule_hits": [infer_feedback_rule_hits(demo.candidate) for demo in demonstrations],
            "prompt_demo_clue_features": [render_sample_clue_text(demo.candidate) for demo in demonstrations],
            "prompt_demo_feedback_rule_text": [render_feedback_rule_text(demo.candidate) for demo in demonstrations],
            "prompt_demo_heuristic_rationales": [render_demo_reason(demo.candidate) for demo in demonstrations],
        }
        prior_key = (query.source, query.api)
        prior_stats = api_prior_stats.get(prior_key, {"support": 0, "positive": 0})
        prior_support = int(prior_stats["support"])
        prior_positive = int(prior_stats["positive"])
        prior_rate = float(prior_positive / prior_support) if prior_support else 0.5
        prediction_row["pool_api_support"] = prior_support
        prediction_row["pool_api_positive_rate"] = prior_rate
        if config.icl.enable_feedback_calibration:
            prediction_row = apply_feedback_calibration(prediction_row, query)

        labels.append(query.label)
        predictions.append(int(prediction_row["predicted_label"]))
        rankings.append(rows[:10])
        prompt_rows_by_query.append(prompt_demo_rows)
        prediction_rows.append(prediction_row)

    metrics = classification_metrics(labels, predictions)
    metrics.update({f"retrieval_{key}": value for key, value in retrieval_metrics(queries, rankings).items()})
    dump_json(metrics, config.output_dir / f"{output_name}_metrics.json")
    dump_json(prediction_rows, config.output_dir / f"{output_name}_predictions.json")
    dump_json(case_studies(queries, rankings, config.icl.max_case_studies), config.output_dir / f"{output_name}_cases.json")
    diagnostics = {
        "source_metrics": _source_metrics(queries, predictions),
        "confusion_counts_by_source": _confusion_counts_by_source(queries, predictions),
        "so_error_buckets": _bucket_so_errors(prediction_rows, queries, prompt_rows_by_query),
    }
    dump_json(diagnostics, config.output_dir / f"{output_name}_diagnostics.json")
    dump_json(
        build_error_analysis(queries, prediction_rows, prompt_rows_by_query),
        config.output_dir / f"{output_name}_error_analysis.json",
    )
    preprocessing_feedback = build_preprocessing_feedback(queries, prediction_rows, prompt_rows_by_query)
    dump_json(preprocessing_feedback, config.output_dir / f"{output_name}_preprocessing_feedback.json")
    export_preprocessing_feedback_csv(
        preprocessing_feedback,
        config.output_dir / f"{output_name}_preprocessing_feedback.csv",
    )
    return metrics


def run_rq1(config: AppConfig) -> dict:
    _, grouped, artifacts = _load_trained_artifacts(config)
    _validate_library(config, grouped)
    selected_libraries = [config.library] if config.library else list(grouped.keys())
    ensure_dir(config.output_dir)

    per_library: dict[str, dict] = {}
    library_metrics = []
    for library in selected_libraries:
        subset_config = _subset_config(config, library)
        samples = grouped[library]
        export_augmented_samples(samples, subset_config.output_dir / f"{library}_augmented_samples.csv")
        folds = build_kfold_splits(samples, config.split.n_splits, config.training.seed)
        if config.rq_max_folds is not None:
            folds = folds[: max(1, config.rq_max_folds)]
        fold_metrics = []
        for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
            pool = [samples[idx] for idx in train_idx]
            queries = [samples[idx] for idx in test_idx]
            fold_metrics.append(run_icl_fold(subset_config, pool, queries, artifacts, f"rq1_fold_{fold_id}"))
        result = {
            "folds": fold_metrics,
            "macro_average": _metric_average(fold_metrics),
        }
        dump_json(result, subset_config.output_dir / "rq1_results.json")
        per_library[library] = result
        library_metrics.append({"library": library, **result["macro_average"]})

    summary = {
        "rq": "RQ1",
        "libraries": selected_libraries,
        "library_metrics": library_metrics,
        "macro_average": _metric_average(library_metrics),
    }
    dump_json(summary, config.output_dir / "rq1_summary.json")
    return {"per_library": per_library, "summary": summary}


def run_rq2(config: AppConfig) -> dict:
    _, grouped, artifacts = _load_trained_artifacts(config)
    _validate_library(config, grouped)
    selected_libraries = [config.library] if config.library else list(grouped.keys())
    ensure_dir(config.output_dir)

    per_library: dict[str, dict] = {}
    for library in selected_libraries:
        subset_config = _subset_config(config, library)
        samples = grouped[library]
        train_samples, _, test_samples = stratified_split(
            samples,
            train_size=config.split.train_size,
            dev_size=config.split.dev_size,
            test_size=config.split.test_size,
            seed=config.training.seed,
        )
        ablations = {}
        for name, lexical, semantic, structural in [
            ("wo_lexical", 0.0, 0.7142857143, 0.2857142857),
            ("wo_semantic", 0.6, 0.0, 0.4),
            ("wo_structural", 0.375, 0.625, 0.0),
        ]:
            ablation_config = deepcopy(subset_config)
            ablation_config.model.weights.lexical = lexical
            ablation_config.model.weights.semantic = semantic
            ablation_config.model.weights.structural = structural
            ablations[name] = run_icl_fold(
                ablation_config,
                train_samples,
                test_samples,
                artifacts,
                f"rq2_{name}",
            )
        no_evidence_config = deepcopy(subset_config)
        no_evidence_config.icl.evidence_augmented = False
        ablations["wo_evidence_augmented"] = run_icl_fold(
            no_evidence_config,
            train_samples,
            test_samples,
            artifacts,
            "rq2_wo_evidence_augmented",
        )
        dump_json(ablations, subset_config.output_dir / "rq2_results.json")
        per_library[library] = ablations

    summary = {"rq": "RQ2", "libraries": selected_libraries}
    dump_json(summary, config.output_dir / "rq2_summary.json")
    return {"per_library": per_library, "summary": summary}


def run_rq3(config: AppConfig) -> dict:
    _, grouped, artifacts = _load_trained_artifacts(config)
    _validate_library(config, grouped)
    selected_libraries = [config.library] if config.library else list(grouped.keys())
    ensure_dir(config.output_dir)

    per_library: dict[str, dict] = {}
    for library in selected_libraries:
        subset_config = _subset_config(config, library)
        samples = grouped[library]
        train_samples, _, test_samples = stratified_split(
            samples,
            train_size=config.split.train_size,
            dev_size=config.split.dev_size,
            test_size=config.split.test_size,
            seed=config.training.seed,
        )
        factor_results = {}
        for top_k in [2, 4, 6, 8]:
            factor_config = deepcopy(subset_config)
            factor_config.icl.top_k = top_k
            factor_results[f"top_k_{top_k}"] = run_icl_fold(
                factor_config,
                train_samples,
                test_samples,
                artifacts,
                f"rq3_top_k_{top_k}",
            )
        for order_strategy in ["nearest_last", "nearest_first"]:
            factor_config = deepcopy(subset_config)
            factor_config.icl.order_strategy = order_strategy
            factor_results[f"order_{order_strategy}"] = run_icl_fold(
                factor_config,
                train_samples,
                test_samples,
                artifacts,
                f"rq3_order_{order_strategy}",
            )
        dump_json(factor_results, subset_config.output_dir / "rq3_results.json")
        per_library[library] = factor_results

    summary = {"rq": "RQ3", "libraries": selected_libraries}
    dump_json(summary, config.output_dir / "rq3_summary.json")
    return {"per_library": per_library, "summary": summary}


def run_rq4(config: AppConfig) -> dict:
    samples, grouped, artifacts = _load_trained_artifacts(config)
    ensure_dir(config.output_dir)
    if not config.rq4_query_library or not config.rq4_pool_library:
        raise ValueError("RQ4 requires rq4_query_library and rq4_pool_library")
    if config.rq4_query_library not in grouped:
        raise ValueError(f"Unknown rq4_query_library: {config.rq4_query_library}")
    if config.rq4_pool_library not in grouped:
        raise ValueError(f"Unknown rq4_pool_library: {config.rq4_pool_library}")

    query_library = config.rq4_query_library
    pool_library = config.rq4_pool_library
    queries = grouped[query_library]
    pool = grouped[pool_library]
    query_language = queries[0].language
    pool_language = pool[0].language
    setting = "within_language" if query_language == pool_language else "cross_language"

    metrics = run_icl_fold(
        config,
        pool,
        queries,
        artifacts,
        f"rq4_query_{query_library}_pool_{pool_library}",
    )
    result = {
        "rq": "RQ4",
        "setting": setting,
        "query_library": query_library,
        "pool_library": pool_library,
        "query_language": query_language,
        "pool_language": pool_language,
        "num_queries": len(queries) if config.icl.max_queries is None else min(len(queries), config.icl.max_queries),
        "num_pool_samples": len(pool),
        "metrics": metrics,
    }
    dump_json(result, config.output_dir / "rq4_results.json")
    return result


def run_rq_experiment(config: AppConfig) -> dict:
    if not config.rq_id:
        raise ValueError("rq command requires rq_id in {'RQ1', 'RQ2', 'RQ3', 'RQ4'}")
    rq_id = config.rq_id.upper()
    if rq_id == "RQ1":
        return run_rq1(config)
    if rq_id == "RQ2":
        return run_rq2(config)
    if rq_id == "RQ3":
        return run_rq3(config)
    if rq_id == "RQ4":
        return run_rq4(config)
    raise ValueError(f"Unknown rq_id: {config.rq_id}")
