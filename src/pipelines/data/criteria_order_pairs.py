def apply_criteria_less_to_higher(target_scores: dict) -> tuple[list[str], str]:
    ranked_t_scores = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
    lowest_score_keys = ranked_t_scores[1:]
    highest_score_key = ranked_t_scores[0]
    return lowest_score_keys, highest_score_key
