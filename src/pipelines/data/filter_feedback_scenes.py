def max_delta_scene_pair(scene_list: list[dict]) -> dict:
    ranked_t_scores = sorted(scene_list, key=lambda x: x["source_score"])
    lowest_score_entry = ranked_t_scores[0]
    return lowest_score_entry
