from __future__ import annotations

import json

from .clues import build_sample_clue_features
from .data import ApiSample


POSITIVE_REASON_PATTERNS = (
    "clear, actionable solution",
    "provides actionable knowledge",
    "actionable guidance",
    "demonstrating correct usage",
    "directly teaching",
    "concrete solution",
    "explicitly teaches",
    "teaches how to use",
    "teaches how to obtain",
)

NEGATIVE_REASON_PATTERNS = (
    "not teaching",
    "not teach",
    "only mentioned",
    "only in passing",
    "incidentally",
    "question",
    "bug report",
    "not providing actionable",
    "does not teach",
    "incidental",
    "background to",
    "not the target api",
)

TU_STRUCTURAL_REASON_PATTERNS = (
    "interface",
    "implementation",
    "represented",
    "defines",
    "characteristics",
    "role",
    "relationship",
    "specializations",
    "property",
    "constructor",
)


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def extract_llm_label_from_row(row: dict) -> int | None:
    if "llm_predicted_label" in row:
        return int(row["llm_predicted_label"])
    raw = str(row.get("raw_response", "")).strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    label = payload.get("label")
    if label in {0, 1}:
        return int(label)
    return None


def infer_feedback_calibration_actions(row: dict, sample: ApiSample) -> list[str]:
    predicted_label = int(row["predicted_label"])
    llm_label = extract_llm_label_from_row(row)
    reason = str(row.get("reason", "")).lower()
    features = build_sample_clue_features(sample)

    demo_labels: list[int] = []
    for raw_label in row.get("prompt_demo_labels", []):
        try:
            parsed = int(raw_label)
        except (TypeError, ValueError):
            continue
        if parsed in {0, 1}:
            demo_labels.append(parsed)
    positive_demo_votes = sum(label == 1 for label in demo_labels)
    negative_demo_votes = len(demo_labels) - positive_demo_votes

    actions: list[str] = []

    if (
        predicted_label == 0
        and llm_label == 1
        and _contains_any(reason, POSITIVE_REASON_PATTERNS)
        and not _contains_any(reason, NEGATIVE_REASON_PATTERNS)
    ):
        actions.append("restore_positive_llm_reason")

    if (
        predicted_label == 1
        and sample.source == "TU"
        and features["likely_list_only"] == "yes"
        and features["solution_like"] == "no"
    ):
        actions.append("demote_tu_list_only")

    if (
        predicted_label == 1
        and sample.source == "TU"
        and features["likely_structural_only"] == "yes"
    ):
        actions.append("demote_tu_structural_only")

    if (
        predicted_label == 1
        and sample.source == "TU"
        and features["concise"] == "yes"
        and _contains_any(reason, TU_STRUCTURAL_REASON_PATTERNS)
    ):
        actions.append("demote_tu_concise_structural_reason")

    if (
        predicted_label == 1
        and _contains_any(reason, NEGATIVE_REASON_PATTERNS)
        and int(features["answer_excerpt_api_mentions"]) == 0
        and features["accepted_answer_like"] == "no"
        and features["final_solution_like"] == "no"
    ):
        actions.append("demote_negative_reason_without_api_evidence")

    if (
        predicted_label == 1
        and sample.source == "SO"
        and features["likely_wrong_api_focus"] == "yes"
        and int(features["answer_excerpt_api_mentions"]) <= 1
        and _contains_any(reason, NEGATIVE_REASON_PATTERNS)
    ):
        actions.append("demote_so_wrong_api_focus")

    if (
        predicted_label == 1
        and sample.source == "SO"
        and features["question_like"] == "yes"
        and features["unresolved_like"] == "yes"
        and int(features["answer_excerpt_api_mentions"]) == 0
        and features["accepted_answer_like"] == "no"
        and features["final_solution_like"] == "no"
    ):
        actions.append("demote_so_question_heavy_without_answer_signal")

    if (
        predicted_label == 1
        and sample.source == "SO"
        and features["solution_like"] == "yes"
        and features["concise"] == "yes"
    ):
        actions.append("demote_so_concise_solution_like")

    if (
        predicted_label == 1
        and sample.source == "SO"
        and _contains_any(reason, POSITIVE_REASON_PATTERNS)
        and features["question_like"] == "no"
        and features["solution_like"] == "no"
        and features["accepted_answer_like"] == "no"
        and features["final_solution_like"] == "no"
    ):
        actions.append("demote_so_positive_reason_without_question_or_solution")

    if (
        predicted_label == 0
        and sample.source == "TU"
        and features["concise"] == "yes"
        and features["api_focus"] in {"medium", "high"}
        and features["code_like"] == "yes"
        and features["likely_list_only"] == "no"
        and features["likely_structural_only"] == "no"
        and _contains_any(reason, POSITIVE_REASON_PATTERNS)
        and not _contains_any(reason, NEGATIVE_REASON_PATTERNS)
        and (features["solution_like"] == "yes" or features["tutorial_structural_like"] == "yes")
    ):
        actions.append("restore_tu_concise_actionable_guidance")

    if (
        sample.source == "TU"
        and features["likely_list_only"] == "yes"
        and features["solution_like"] == "yes"
        and _contains_any(reason, POSITIVE_REASON_PATTERNS)
    ):
        actions.append("restore_tu_list_usage_reason")

    if (
        predicted_label == 0
        and sample.source == "TU"
        and features["concise"] == "yes"
        and features["tutorial_list_like"] == "yes"
        and features["tutorial_structural_like"] == "yes"
    ):
        actions.append("restore_tu_concise_list_structural_guidance")

    if (
        predicted_label == 0
        and sample.source == "SO"
        and (
            llm_label == 1
            or positive_demo_votes >= max(3, negative_demo_votes + 1)
        )
        and features["solution_like"] == "yes"
        and int(features["answer_excerpt_api_mentions"]) >= 1
        and features["likely_wrong_api_focus"] == "no"
        and features["likely_general_advice"] == "no"
    ):
        actions.append("restore_so_solution_with_api_signal")

    if (
        predicted_label == 1
        and sample.source == "SO"
        and _contains_any(reason, POSITIVE_REASON_PATTERNS)
        and _contains_any(reason, NEGATIVE_REASON_PATTERNS)
        and int(features["answer_excerpt_api_mentions"]) <= 2
    ):
        actions.append("demote_so_mixed_reason_low_answer_signal")

    return actions


def apply_feedback_calibration(row: dict, sample: ApiSample) -> dict:
    updated = dict(row)
    actions = infer_feedback_calibration_actions(updated, sample)
    updated["feedback_calibration_actions"] = actions
    updated["feedback_calibration_applied"] = bool(actions)
    updated["pre_feedback_predicted_label"] = int(updated["predicted_label"])

    label = int(updated["predicted_label"])
    if "restore_positive_llm_reason" in actions:
        label = 1
    if "demote_tu_list_only" in actions:
        label = 0
    if "demote_tu_structural_only" in actions:
        label = 0
    if "demote_tu_concise_structural_reason" in actions:
        label = 0
    if "demote_negative_reason_without_api_evidence" in actions:
        label = 0
    if "demote_so_wrong_api_focus" in actions:
        label = 0
    if "demote_so_question_heavy_without_answer_signal" in actions:
        label = 0
    if "demote_so_concise_solution_like" in actions:
        label = 0
    if "demote_so_positive_reason_without_question_or_solution" in actions:
        label = 0
    if "demote_so_mixed_reason_low_answer_signal" in actions:
        label = 0
    if "restore_tu_concise_actionable_guidance" in actions:
        label = 1
    if "restore_tu_list_usage_reason" in actions:
        label = 1
    if "restore_tu_concise_list_structural_guidance" in actions:
        label = 1
    if "restore_so_solution_with_api_signal" in actions:
        label = 1
    updated["predicted_label"] = label
    return updated


def should_verify_so_positive(row: dict, sample: ApiSample) -> tuple[bool, list[str]]:
    if sample.source != "SO" or int(row.get("predicted_label", 0)) != 1:
        return False, []

    features = build_sample_clue_features(sample)
    reason = str(row.get("reason", "")).lower()
    risk_flags: list[str] = []

    if features["question_like"] == "yes":
        risk_flags.append("question_like")
    if features["unresolved_like"] == "yes" or features["likely_unsolved_qa"] == "yes":
        risk_flags.append("unresolved_like")
    if int(features["answer_excerpt_api_mentions"]) <= 2:
        risk_flags.append("low_answer_api_signal")
    if _contains_any(reason, NEGATIVE_REASON_PATTERNS):
        risk_flags.append("negative_reason")

    should_verify = (
        "question_like" in risk_flags
        and "low_answer_api_signal" in risk_flags
        and "negative_reason" in risk_flags
    )
    return should_verify, risk_flags
