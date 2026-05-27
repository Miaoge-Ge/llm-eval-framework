from llm_eval.ifeval import evaluate


def test_ifeval_no_comma_passes_when_followed():
    report = evaluate("Write a note.", "no commas here at all", ["punctuation:no_comma"], [{}])
    assert report["prompt_strict"] is True
    assert report["instructions_followed_strict"] == 1


def test_ifeval_detects_violation():
    report = evaluate("Write a note.", "this, sentence, has commas", ["punctuation:no_comma"], [{}])
    assert report["prompt_strict"] is False
    assert report["instructions_followed_strict"] == 0


def test_ifeval_requires_all_instructions_for_prompt_strict():
    instruction_ids = ["change_case:english_lowercase", "punctuation:no_comma"]
    followed = evaluate("x", "all lowercase and no commas", instruction_ids, [{}, {}])
    assert followed["instructions_total"] == 2
    assert followed["prompt_strict"] is True

    partial = evaluate("x", "Has Caps but no commas", instruction_ids, [{}, {}])
    assert partial["instructions_followed_strict"] == 1
    assert partial["prompt_strict"] is False
