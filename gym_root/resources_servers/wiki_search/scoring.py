# Adapted from Search-R1's verl/utils/reward_score/qa_em.py (github.com/PeterGriffinJin/Search-R1;
# Copyright 2024 Bytedance Ltd. and/or its affiliates; Apache License 2.0): normalize_answer,
# em_check and extract_solution. See NOTICE.
import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the answer from the solution string."""

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are no matches, return None
    if len(matches) == 0:
        return None

    # Return the last answer tag content
    return matches[-1].group(1).strip()
