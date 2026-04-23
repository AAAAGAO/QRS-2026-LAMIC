from __future__ import annotations

from collections import Counter

from .clues import FEEDBACK_RULE_DESCRIPTIONS, infer_feedback_rule_hits
from .data import ApiSample


def _reason_text(row: dict) -> str:
    return str(row.get("reason", "")).lower()


def _bucket_so_false_positive(query: ApiSample, row: dict, prompt_rows: list[dict]) -> tuple[str, str]:
    reason = _reason_text(row)
    prompt_apis = [str(item.get("api", "")) for item in prompt_rows]
    if any(token in reason for token in ["question", "bug", "error", "exception", "symptom", "help"]):
        return (
            "weak_answer_or_unresolved_qa",
            "Add stronger negative clues for unresolved questions, symptom descriptions, and answers that do not actually solve the issue.",
        )
    if any(token in reason for token in ["another api", "other api", "wrong api", "focuses on", "not the target api"]):
        return (
            "wrong_api_focus",
            "Add cues that down-rank answers whose useful content focuses on a nearby Joda-Time API rather than the target API.",
        )
    if any(api and api != query.api for api in prompt_apis):
        return (
            "wrong_api_focus",
            "Add cues that down-rank answers whose useful content focuses on a nearby Joda-Time API rather than the target API.",
        )
    return (
        "general_so_false_positive",
        "Inspect whether the SO prompt still treats generic date-time advice as target-API guidance.",
    )


def _bucket_so_false_negative(query: ApiSample, row: dict, prompt_rows: list[dict]) -> tuple[str, str]:
    reason = _reason_text(row)
    if len(query.fragment) <= 900:
        return (
            "concise_solution_missed",
            "Add positive cues for short but complete code recipes and concise accepted-answer style fixes.",
        )
    if any(token in reason for token in ["not specifically", "indirectly", "incidental", "not the target api"]):
        return (
            "over_strict_api_focus",
            "Add cues that allow compact but still target-relevant guidance even when other APIs appear in the same answer.",
        )
    if any(token in reason for token in ["question", "rant", "requests help", "symptom"]):
        return (
            "question_penalty_overfire",
            "Do not punish SO samples just because they begin with a question if the answer portion contains a reusable fix.",
        )
    return (
        "general_so_false_negative",
        "Review missed SO positives for missing clues about solved Q&A structure and target-API utility.",
    )


def _bucket_tu_false_positive(row: dict) -> tuple[str, str]:
    reason = _reason_text(row)
    if any(token in reason for token in ["list", "completeness", "listed", "example"]):
        return (
            "list_or_example_only",
            "Add negative tutorial clues for APIs that are only listed or used illustratively.",
        )
    if any(token in reason for token in ["structural", "javadoc", "static", "constants"]):
        return (
            "structural_fact_only",
            "Add negative tutorial clues for structural facts and Javadoc-like information.",
        )
    return (
        "general_tu_false_positive",
        "Review tutorial false positives for weak information-density and incidental API mentions.",
    )


def _bucket_tu_false_negative(query: ApiSample, row: dict) -> tuple[str, str]:
    reason = _reason_text(row)
    if len(query.fragment) <= 900:
        return (
            "concise_tutorial_guidance_missed",
            "Add positive tutorial clues for short but actionable usage guidance.",
        )
    if "unsure" in reason:
        return (
            "uncertainty_overfire",
            "Tighten the uncertainty rule so clear tutorial usage guidance is not rejected too aggressively.",
        )
    return (
        "general_tu_false_negative",
        "Review tutorial false negatives for under-recognized usage explanations.",
    )


def _summarize_sample(query: ApiSample, row: dict, bucket: str) -> dict[str, str | int]:
    return {
        "sample_id": query.sample_id,
        "source": query.source,
        "dataset": query.dataset,
        "api": query.api,
        "gold_label": query.label,
        "predicted_label": int(row["predicted_label"]),
        "bucket": bucket,
        "feedback_rule_hits": infer_feedback_rule_hits(query),
        "reason": str(row.get("reason", "")),
        "fragment_preview": query.fragment[:500],
    }


def build_error_analysis(
    queries: list[ApiSample],
    prediction_rows: list[dict],
    prompt_rows_by_query: list[list[dict]],
) -> dict:
    by_id = {query.sample_id: query for query in queries}
    grouped_samples: dict[str, dict[str, list[dict[str, str | int]]]] = {
        "false_positives": {},
        "false_negatives": {},
    }
    clue_counter: Counter[tuple[str, str]] = Counter()
    feedback_rule_counter: Counter[str] = Counter()

    for row, prompt_rows in zip(prediction_rows, prompt_rows_by_query, strict=True):
        query = by_id[int(row["sample_id"])]
        gold = int(row["gold_label"])
        predicted = int(row["predicted_label"])
        if gold == predicted:
            continue

        if query.source == "SO":
            if gold == 0 and predicted == 1:
                bucket, suggestion = _bucket_so_false_positive(query, row, prompt_rows)
                group = "false_positives"
            else:
                bucket, suggestion = _bucket_so_false_negative(query, row, prompt_rows)
                group = "false_negatives"
        else:
            if gold == 0 and predicted == 1:
                bucket, suggestion = _bucket_tu_false_positive(row)
                group = "false_positives"
            else:
                bucket, suggestion = _bucket_tu_false_negative(query, row)
                group = "false_negatives"

        grouped_samples[group].setdefault(bucket, []).append(_summarize_sample(query, row, bucket))
        clue_counter[(bucket, suggestion)] += 1
        feedback_rule_counter.update(infer_feedback_rule_hits(query))

    augmentation_clues = [
        {
            "bucket": bucket,
            "count": count,
            "suggestion": suggestion,
        }
        for (bucket, suggestion), count in clue_counter.most_common()
    ]
    feedback_rules = [
        {
            "rule_id": rule_id,
            "count": count,
            "description": FEEDBACK_RULE_DESCRIPTIONS.get(rule_id, ""),
        }
        for rule_id, count in feedback_rule_counter.most_common()
    ]

    return {
        "summary": {
            "num_false_positives": sum(len(items) for items in grouped_samples["false_positives"].values()),
            "num_false_negatives": sum(len(items) for items in grouped_samples["false_negatives"].values()),
        },
        "buckets": grouped_samples,
        "augmentation_clues": augmentation_clues,
        "feedback_rules": feedback_rules,
    }
