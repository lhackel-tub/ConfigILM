from math import isclose as fequal

from torch import Tensor

from configvlm import util


def test_indent_simple():
    test_string = "test 123 test"
    for n in range(5):
        assert (
            util._indent(test_string, num_spaces=n, indent_first=True)
            == " " * n + test_string
        )


def test_indent_multiline():
    test_string = "this is a\ntest string"
    expected = "this is a\n    test string"
    assert util._indent(test_string, num_spaces=4) == expected


def test_indent_multiline_first():
    test_string = "this is a\ntest string"
    expected = "    this is a\n    test string"
    assert util._indent(test_string, num_spaces=4, indent_first=True) == expected


def test_indent_multiline_first_lstrip():
    test_string = "  this is a\n   test string"
    expected = "    this is a\n    test string"
    assert util._indent(test_string, num_spaces=4, indent_first=True) == expected


def test_indent_multiline_first_no_rstrip():
    test_string = "  this is a    \n   test string  "
    expected = "    this is a\n    test string"
    assert util._indent(test_string, num_spaces=4, indent_first=True) != expected


def test_round_to():
    x = 123.456789
    # fequal -> float approximate for equal
    assert fequal(util.round_to(x, 10), 120)
    assert fequal(util.round_to(x, 2), 124)
    assert fequal(util.round_to(x, 11), 121)
    assert fequal(util.round_to(x, 0.02), 123.46)
    assert fequal(util.round_to(x, 0.123), 123.492)  # 1004 * 0.123 is closer than 1003


def test_convert():
    lbl_tensor = Tensor(
        [[0, 1, 0, 0, 1], [0, 1, 1, 0, 1], [0, 0, 1, 0, 0], [1, 0, 0, 1, 0]]
    )
    logits_tensor = Tensor(
        [
            [0.7012, 0.9606, 0.2721, 0.2611, 0.0460],
            [0.4947, 0.7894, 0.2541, 0.1715, 0.1386],
            [0.9105, 0.3274, 0.0637, 0.8683, 0.8432],
            [0.3140, 0.2533, 0.4329, 0.0394, 0.0276],
        ]
    )
    # labels to index and then flatten
    expected_lbl = [1, 4, 1, 2, 4, 2, 0, 3]
    # for each label the respective tensor x times, once per true class
    # e.g. if in tensor t1 there are 2 classes positive expected, then t1 has to be
    # present 2 times
    expected_logits = [
        [0.7012, 0.9606, 0.2721, 0.2611, 0.0460],  # t1 2 times (clases 1, 4)
        [0.7012, 0.9606, 0.2721, 0.2611, 0.0460],
        [0.4947, 0.7894, 0.2541, 0.1715, 0.1386],  # t2 3 times (classes 1, 2, 4)
        [0.4947, 0.7894, 0.2541, 0.1715, 0.1386],
        [0.4947, 0.7894, 0.2541, 0.1715, 0.1386],
        [0.9105, 0.3274, 0.0637, 0.8683, 0.8432],  # t3 1 time (for class 2)
        [0.3140, 0.2533, 0.4329, 0.0394, 0.0276],  # t4 2 times (classes 0, 3)
        [0.3140, 0.2533, 0.4329, 0.0394, 0.0276],
    ]
    actual_lbl, actual_logits = util.convert(lbl_tensor, logits_tensor)

    abs_tol = 5e-8
    assert len(actual_lbl) == len(expected_lbl)
    assert actual_lbl == expected_lbl
    assert len(actual_logits) == len(expected_logits)
    for a, e in zip(actual_logits, expected_logits):
        assert len(a) == len(e)
        for av, ev in zip(a, e):
            assert fequal(av, ev, abs_tol=abs_tol), (
                f"{av} is not the same as {ev}:\n" f"diff: {abs(av-ev)} > {abs_tol}"
            )


def test_avg_meter():
    avg_mtr = util.AverageMeter(name="Testmeter", fmt=":.2f")
    assert avg_mtr.avg == 0
    assert avg_mtr.sum == 0
    assert avg_mtr.count == 0
    assert avg_mtr.val == 0

    avg_mtr.update(42)
    assert avg_mtr.avg == 42
    assert avg_mtr.sum == 42
    assert avg_mtr.count == 1
    assert avg_mtr.val == 42

    avg_mtr.update(0)
    assert avg_mtr.avg == 21
    assert avg_mtr.sum == 42
    assert avg_mtr.count == 2
    assert avg_mtr.val == 0

    avg_mtr.update(4, n=2)
    assert avg_mtr.avg == (42 + 4 + 4) / 4
    assert avg_mtr.sum == 42 + 4 + 4
    assert avg_mtr.count == 4
    assert avg_mtr.val == 4

    assert str(avg_mtr) == f"Testmeter {4:.2f} ({(42+4+4)/4:.2f})"


def test_obj_size():
    # for each constant 28
    # for each space 2, but always 2 at the same time, but not for first one
    assert util.get_obj_size([]) == 56
    assert util.get_obj_size([1]) == 56 + 8 + 28
    assert util.get_obj_size([1, 2]) == 56 + 8 * 2 + 28 * 2
    assert util.get_obj_size([1, 2, 3]) == 56 + 8 * 4 + 28 * 3
    assert util.get_obj_size([1, 2, 3, 4]) == 56 + 8 * 4 + 28 * 4
    assert util.get_obj_size([1, 2, 3, 4, 5]) == 56 + 8 * 6 + 28 * 5

    # same constant repeated = no extra space used
    assert util.get_obj_size([1, 2, 3, 4, 5, 5]) == 56 + 8 * 6 + 28 * 5

    # appending works in different increments
    # The growth pattern is:  0, 4, 8, 16, 25, 35, 46, 58, 72, 88, ...
    # https://stackoverflow.com/questions/7247298/size-of-list-in-memory
    x = list()
    x.append(1)
    assert util.get_obj_size(x) == 56 + 8 * 4 + 28
    for i in [2, 3, 4]:
        x.append(i)
    assert util.get_obj_size(x) == 56 + 8 * 4 + 28 * 4
