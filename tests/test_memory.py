"""Tiny hand-check tests mirroring the video (run with pytest or python)."""
from src.memory import keep_from_step, run_sequence, run_selective, MultiSlotMemory, FOUR_SLOT_ALPHAS
from src.games import predict_fixed, predict_selective, token_value
from src.benchmarks import transformer_numbers_per_layer, mamba_numbers_per_layer, ratio_at


def test_hand_rule_4_0_0_2():
    assert run_sequence([4, 0, 0, 2], keep=0.5) == [2.0, 1.0, 0.5, 1.25]


def test_exercise_2_0_6():
    assert run_sequence([2, 0, 6], keep=0.5) == [1.0, 0.5, 3.25]


def test_keep_step_duality():
    assert abs(keep_from_step(0.01) - 0.9900498) < 1e-6
    assert keep_from_step(5) < 0.01
    assert keep_from_step(20) < 1e-8


def test_row_K3BQ8ZZ_gives_8():
    # K->0? K=(10)%10=0, 3->3, B->1, Q->6, 8->8, Z->5,Z->5 ; last digit 8 must survive
    assert predict_selective(list("K3BQ8ZZ")) == 8


def test_memory_scale_ratios():
    assert transformer_numbers_per_layer(1000) == 2 * 768 * 1000
    assert mamba_numbers_per_layer() == 1536 * 16 == 24576
    assert abs(ratio_at(1000) - 62.5) < 1e-9
    assert abs(ratio_at(32000) - 2000.0) < 1e-9
