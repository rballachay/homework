from code import (
    build_current_surrounding_pairs,
    expand_surrounding_words,
    compute_topk_similar,
)
import torch


def t_est_build_current_surrounding_pairs():
    text = "dogs and cats are playing".split()
    surroundings, currents = build_current_surrounding_pairs(text, window_size=1)
    assert surroundings == [["dogs", "cats"], ["and", "are"], ["cats", "playing"]]
    assert currents == ["and", "cats", "are"]

    indices = [110, 3, 4887, 11, 31]
    surroundings, currents = build_current_surrounding_pairs(indices, window_size=1)
    assert currents == [3, 4887, 11]
    assert surroundings == [[110, 4887], [3, 11], [4887, 31]]


def t_est_expand_surrounding_words():
    surroundings = [["dogs", "cats"], ["and", "are"], ["cats", "playing"]]
    currents = ["and", "cats", "are"]
    surrounding_expanded, current_expanded = expand_surrounding_words(
        surroundings, currents
    )

    assert surrounding_expanded == ["dogs", "cats", "and", "are", "cats", "playing"]
    assert current_expanded == ["and", "and", "cats", "cats", "are", "are"]


def test_compute_topk_similar():

    word = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.1]])
    emb = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.1], [0, -0.1, -0.1, 0, 0], [0, -0.1, -0.1, 0, 0]]
    )
    results = compute_topk_similar(word, emb, 1)
    assert results == [0]
