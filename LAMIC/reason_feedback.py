from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .clues import (
    FEEDBACK_RULE_DESCRIPTIONS,
    count_target_api_mentions,
    extract_answer_focused_excerpt,
    infer_feedback_rule_hits,
    render_feedback_rule_text,
)
from .data import ApiSample
from .error_analysis import build_error_analysis


PREPROCESS_ACTIONS_BY_BUCKET = {
    "weak_answer_or_unresolved_qa": [
        "extract_answer_focused_excerpt",
        "downweight_question_only_lead",
        "require_reusable_fix_signal",
    ],
    "wrong_api_focus": [
        "mark_non_target_api_focus",
        "downweight_incidental_target_api_mentions",
        "highlight_target_api_span_only",
    ],
    "general_so_false_positive": [
        "downweight_generic_datetime_advice",
        "mark_non_target_api_focus",
        "prefer_answer_paragraphs_over_question",
    ],
    "question_penalty_overfire": [
        "preserve_answer_after_question",
        "avoid_question_prefix_penalty",
        "highlight_reusable_fix_paragraph",
    ],
    "general_so_false_negative": [
        "highlight_solution_paragraphs",
        "preserve_multi_api_solution_context",
    ],
    "over_strict_api_focus": [
        "allow_supporting_nearby_api_context",
        "preserve_solution_paragraphs_with_target_api_mentions",
    ],
    "concise_solution_missed": [
        "boost_short_code_recipe",
        "preserve_compact_answer_snippet",
    ],
    "list_or_example_only": [
        "mark_list_context",
        "downweight_example_only_mentions",
        "require_usage_signal_for_positive",
    ],
    "structural_fact_only": [
        "mark_structural_definition",
        "downweight_javadoc_style_description",
    ],
    "general_tu_false_positive": [
        "require_usage_verb_or_task_context",
        "downweight_incidental_api_mentions",
    ],
    "concise_tutorial_guidance_missed": [
        "boost_short_usage_snippet",
        "preserve_single_paragraph_howto",
    ],
    "general_tu_false_negative": [
        "preserve_brief_usage_guidance",
        "boost_task_or_howto_sentences",
    ],
    "uncertainty_overfire": [
        "avoid_default_negative_when_usage_signal_exists",
    ],
}
PREPROCESS_ACTIONS_BY_FEEDBACK_RULE = {
    "so_unresolved_or_help_request": [
        "extract_answer_focused_excerpt",
        "downweight_question_only_lead",
        "require_reusable_fix_signal",
    ],
    "so_target_api_incidental": [
        "mark_non_target_api_focus",
        "downweight_incidental_target_api_mentions",
        "highlight_target_api_span_only",
    ],
    "so_generic_datetime_advice": [
        "downweight_generic_datetime_advice",
        "mark_non_target_api_focus",
        "prefer_answer_paragraphs_over_question",
    ],
    "so_question_with_reusable_fix": [
        "preserve_answer_after_question",
        "avoid_question_prefix_penalty",
        "highlight_reusable_fix_paragraph",
    ],
    "so_answer_quality_signal": [
        "highlight_solution_paragraphs",
        "preserve_multi_api_solution_context",
    ],
    "tu_list_or_example_only": [
        "mark_list_context",
        "downweight_example_only_mentions",
        "require_usage_signal_for_positive",
    ],
    "tu_structural_fact_only": [
        "mark_structural_definition",
        "downweight_javadoc_style_description",
    ],
    "tu_concise_usage_guidance": [
        "boost_short_usage_snippet",
        "preserve_single_paragraph_howto",
    ],
}
ACTION_VERB_PATTERNS = (
    "use ",
    "call ",
    "create ",
    "convert ",
    "parse ",
    "format ",
    "compute ",
    "set ",
    "get ",
    "return ",
    "should ",
    "can ",
)


def _actions_for_bucket(bucket: str) -> list[str]:
    return PREPROCESS_ACTIONS_BY_BUCKET.get(bucket, ["manual_review"])


def _paragraphs(fragment: str) -> list[str]:
    parts = [part.strip() for part in fragment.split("\n\n") if part.strip()]
    return parts or [fragment.strip()]


def _contains_action_verb(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in ACTION_VERB_PATTERNS)


def infer_preprocess_actions(sample: ApiSample) -> list[str]:
    actions: list[str] = []
    for rule_id in infer_feedback_rule_hits(sample):
        actions.extend(PREPROCESS_ACTIONS_BY_FEEDBACK_RULE.get(rule_id, []))
    # Preserve order but deduplicate.
    return list(dict.fromkeys(actions))


def propose_preprocessed_fragment(sample: ApiSample, actions: list[str]) -> str:
    paragraphs = _paragraphs(sample.fragment)
    target_api_paragraphs = [
        paragraph for paragraph in paragraphs if count_target_api_mentions(sample.api, paragraph) >= 1
    ]

    if sample.source == "SO":
        answer_excerpt = extract_answer_focused_excerpt(sample)
        if any(
            action in actions
            for action in (
                "extract_answer_focused_excerpt",
                "prefer_answer_paragraphs_over_question",
                "highlight_reusable_fix_paragraph",
                "preserve_answer_after_question",
            )
        ):
            if answer_excerpt:
                return answer_excerpt
        if any(
            action in actions
            for action in (
                "mark_non_target_api_focus",
                "highlight_target_api_span_only",
                "downweight_generic_datetime_advice",
            )
        ) and target_api_paragraphs:
            return "\n\n".join(target_api_paragraphs[:2])[:1600]
        return sample.fragment[:1600]

    filtered = [
        paragraph
        for paragraph in paragraphs
        if count_target_api_mentions(sample.api, paragraph) >= 1
        and (_contains_action_verb(paragraph) or "```" in paragraph or "    " in paragraph)
    ]
    if filtered:
        return "\n\n".join(filtered[:2])[:1600]
    if target_api_paragraphs:
        return "\n\n".join(target_api_paragraphs[:2])[:1600]
    return sample.fragment[:1600]


def build_focus_fragment(sample: ApiSample) -> str:
    return propose_preprocessed_fragment(sample, infer_preprocess_actions(sample))


def build_preprocessing_feedback(
    queries: list[ApiSample],
    prediction_rows: list[dict],
    prompt_rows_by_query: list[list[dict]],
) -> dict:
    analysis = build_error_analysis(queries, prediction_rows, prompt_rows_by_query)
    by_id = {query.sample_id: query for query in queries}
    sample_feedback: list[dict[str, object]] = []
    action_counter: Counter[str] = Counter()

    for group_name, buckets in analysis["buckets"].items():
        for bucket, items in buckets.items():
            actions = _actions_for_bucket(bucket)
            for item in items:
                sample = by_id[int(item["sample_id"])]
                feedback_rule_hits = infer_feedback_rule_hits(sample)
                row = {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "dataset": sample.dataset,
                    "api": sample.api,
                    "gold_label": sample.label,
                    "predicted_label": int(item["predicted_label"]),
                    "error_group": group_name,
                    "error_bucket": bucket,
                    "reason": str(item["reason"]),
                    "feedback_rule_hits": feedback_rule_hits,
                    "feedback_rule_text": render_feedback_rule_text(sample),
                    "preprocess_actions": actions,
                    "preprocess_action_text": "; ".join(actions),
                    "proposed_preprocessed_fragment": propose_preprocessed_fragment(sample, actions),
                    "fragment_preview": sample.fragment[:500],
                }
                sample_feedback.append(row)
                action_counter.update(actions)

    preprocessing_rules = [
        {
            "action": action,
            "count": count,
        }
        for action, count in action_counter.most_common()
    ]

    return {
        "summary": analysis["summary"],
        "augmentation_clues": analysis["augmentation_clues"],
        "feedback_rules": analysis.get("feedback_rules", []),
        "preprocessing_rules": preprocessing_rules,
        "sample_feedback": sample_feedback,
    }


def export_preprocessing_feedback_csv(feedback: dict, path: str | Path) -> None:
    rows = []
    for row in feedback.get("sample_feedback", []):
        rows.append(
            {
                **row,
                "feedback_rule_hits": "|".join(row.get("feedback_rule_hits", [])),
                "preprocess_actions": "|".join(row.get("preprocess_actions", [])),
            }
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
