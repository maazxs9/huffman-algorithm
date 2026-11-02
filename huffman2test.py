from __future__ import annotations

from random import shuffle

import pytest
from hypothesis import given, assume, settings
from hypothesis.strategies import binary, integers, dictionaries, text

from compress import *
# from compress import _get_frequency, _collect_elements

settings.register_profile("norand", settings(derandomize=True, max_examples=200))
settings.load_profile("norand")


# === Test Byte Utilities ===
# Technically, these utilities are given to you in the starter code, so these
# first 3 tests below are just intended as a sanity check to make sure that you
# did not modify these methods and are therefore using them incorrectly.
# You will not be submitting utils.py anyway, so these first three tests are
# solely for your own benefit, as a sanity check.

@given(integers(0, 255))
def test_byte_to_bits(b: int) -> None:
    """ Test that byte_to_bits produces binary strings of length 8."""
    assert set(byte_to_bits(b)).issubset({"0", "1"})
    assert len(byte_to_bits(b)) == 8


@given(text(["0", "1"], min_size=0, max_size=8))
def test_bits_to_byte(s: str) -> None:
    """ Test that bits_to_byte produces a byte."""
    b = bits_to_byte(s)
    assert isinstance(b, int)
    assert 0 <= b <= 255


@given(integers(0, 255), integers(0, 7))
def test_get_bit(byte: int, bit_pos: int) -> None:
    """ Test that get_bit(byte, bit) produces  bit values."""
    b = get_bit(byte, bit_pos)
    assert isinstance(b, int)
    assert 0 <= b <= 1


# === Test the compression code ===

@given(binary(min_size=0, max_size=1000))
def test_build_frequency_dict(byte_list: bytes) -> None:
    """ Test that build_frequency_dict returns dictionary whose values sum up
    to the number of bytes consumed.
    """
    # creates a copy of byte_list, just in case your implementation of
    # build_frequency_dict modifies the byte_list
    b, d = byte_list, build_frequency_dict(byte_list)
    assert isinstance(d, dict)
    assert sum(d.values()) == len(b)


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_build_huffman_tree(d: dict[int, int]) -> None:
    """ Test that build_huffman_tree returns a non-leaf HuffmanTree."""
    t = build_huffman_tree(d)
    assert isinstance(t, HuffmanTree)
    assert not t.is_leaf()


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_get_codes(d: dict[int, int]) -> None:
    """ Test that the sum of len(code) * freq_dict[code] is optimal, so it
    must be invariant under permutation of the dictionary.
    Note: This also tests build_huffman_tree indirectly.
    """
    t = build_huffman_tree(d)
    c1 = get_codes(t)
    d2 = list(d.items())
    shuffle(d2)
    d2 = dict(d2)
    t2 = build_huffman_tree(d2)
    c2 = get_codes(t2)
    assert sum([d[k] * len(c1[k]) for k in d]) == \
           sum([d2[k] * len(c2[k]) for k in d2])


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_number_nodes(d: dict[int, int]) -> None:
    """ If the root is an interior node, it must be numbered two less than the
    number of symbols, since a complete tree has one fewer interior nodes than
    it has leaves, and we are numbering from 0.
    Note: this also tests build_huffman_tree indirectly.
    """
    t = build_huffman_tree(d)
    assume(not t.is_leaf())
    count = len(d)
    number_nodes(t)
    assert count == t.number + 2


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_avg_length(d: dict[int, int]) -> None:
    """ Test that avg_length returns a float in the interval [0, 8], if the max
    number of symbols is 256.
    """
    t = build_huffman_tree(d)
    f = avg_length(t, d)
    assert isinstance(f, float)
    assert 0 <= f <= 8.0


@given(binary(min_size=2, max_size=1000))
def test_compress_bytes(b: bytes) -> None:
    """ Test that compress_bytes returns a bytes object that is no longer
    than the input bytes. Also, the size of the compressed object should be
    invariant under permuting the input.
    Note: this also indirectly tests build_frequency_dict, build_huffman_tree,
    and get_codes.
    """
    d = build_frequency_dict(b)
    t = build_huffman_tree(d)
    c = get_codes(t)
    compressed = compress_bytes(b, c)
    assert isinstance(compressed, bytes)
    assert len(compressed) <= len(b)
    lst = list(b)
    shuffle(lst)
    b = bytes(lst)
    d = build_frequency_dict(b)
    t = build_huffman_tree(d)
    c = get_codes(t)
    compressed2 = compress_bytes(b, c)
    assert len(compressed2) == len(compressed)


@given(binary(min_size=2, max_size=1000))
def test_tree_to_bytes(b: bytes) -> None:
    """ Test that tree_to_bytes generates a bytes representation of a postorder
    traversal of a tree's internal nodes.
    Since each internal node requires 4 bytes to represent, and there are
    1 fewer internal nodes than distinct symbols, the length of the bytes
    produced should be 4 times the length of the frequency dictionary, minus 4.
    Note: also indirectly tests build_frequency_dict, build_huffman_tree, and
    number_nodes.
    """
    d = build_frequency_dict(b)
    assume(len(d) > 1)
    t = build_huffman_tree(d)
    number_nodes(t)
    output_bytes = tree_to_bytes(t)
    dictionary_length = len(d)
    leaf_count = dictionary_length
    assert (4 * (leaf_count - 1)) == len(output_bytes)


# === Test a roundtrip conversion

@given(binary(min_size=1, max_size=1000))
def test_round_trip_compress_bytes(b: bytes) -> None:
    """ Test that applying compress_bytes and then decompress_bytes
    will produce the original text.
    """
    text = b
    freq = build_frequency_dict(text)
    assume(len(freq) > 1)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    compressed = compress_bytes(text, codes)
    decompressed = decompress_bytes(tree, compressed, len(text))
    assert text == decompressed


def test_build_my_huffman():

    # one unique character case
    my_bytes = bytes([65, 65, 65, 65, 65, 65])
    freq_dict = build_frequency_dict(my_bytes)
    tree = build_huffman_tree(freq_dict)
    assert tree.right.symbol != tree.left.symbol
    # assert _get_frequency(tree, freq_dict) == 6

    # even num of unique characters with same frequency
    my_bytes = bytes([65, 67, 65, 68, 69, 67, 68, 71, 72, 69, 71, 83, 84, 72, 83, 84])
    freq_dict = build_frequency_dict(my_bytes)
    tree = build_huffman_tree(freq_dict)

    red = HuffmanTree(None, HuffmanTree(65), HuffmanTree(67))
    blue = HuffmanTree(None, HuffmanTree(68), HuffmanTree(69))
    orange = HuffmanTree(None, red, blue)

    green = HuffmanTree(None, HuffmanTree(71), HuffmanTree(72))
    yellow = HuffmanTree(None, HuffmanTree(83), HuffmanTree(84))
    pink = HuffmanTree(None, green, yellow)

    actual_tree = HuffmanTree(None, orange, pink)

    assert tree == actual_tree

    # assert _get_frequency(tree, freq_dict) == 16
    # assert _get_frequency(tree.left, freq_dict) == _get_frequency(tree.right, freq_dict)

    my_bytes = bytes([65, 67, 65, 68, 69, 67, 68, 71, 72, 69, 71, 84, 72, 84])
    freq_dict = build_frequency_dict(my_bytes)
    tree = build_huffman_tree(freq_dict)

    red = HuffmanTree(None, HuffmanTree(65), HuffmanTree(67))
    orange = HuffmanTree(84)
    yellow = HuffmanTree(None, orange, red)

    blue = HuffmanTree(None, HuffmanTree(68), HuffmanTree(69))
    green = HuffmanTree(None, HuffmanTree(71), HuffmanTree(72))
    pink = HuffmanTree(None, blue, green)

    actual_tree = HuffmanTree(None, yellow, pink)

    assert tree == actual_tree

    # assert _get_frequency(tree, freq_dict) == 14

    # assert _get_frequency(tree.left, freq_dict) < _get_frequency(tree.right, freq_dict)

    my_bytes = bytes([60, 61, 62, 62, 61, 63, 64, 62, 63, 64, 63, 65, 63, 64, 65, 64, 65, 64, 65, 65, 65])
    freq_dict = build_frequency_dict(my_bytes)
    tree = build_huffman_tree(freq_dict)

    b_right = HuffmanTree(None, HuffmanTree(60), HuffmanTree(61))
    b_right2 = HuffmanTree(None, HuffmanTree(62), b_right)
    b_right3 = HuffmanTree(None, HuffmanTree(65), b_right2)
    left = HuffmanTree(None, HuffmanTree(63), HuffmanTree(64))
    actual_tree = HuffmanTree(None, left, b_right3)

    assert tree == actual_tree
    # assert _get_frequency(tree, freq_dict) == sum([value for key, value in freq_dict.items()])
    # assert _get_frequency(tree.left, freq_dict) < _get_frequency(tree.right, freq_dict)


def test_get_my_codes() -> None:

    my_bytes = bytes([65, 65, 65, 65, 65, 65])
    freq_dict = build_frequency_dict(my_bytes)
    tree = build_huffman_tree(freq_dict)
    codes = get_codes(tree)
    assert len(codes) == 2

    my_bytes = bytes([65, 67, 65, 68, 69, 67, 68, 71, 72, 69, 71, 83, 84, 72, 83, 84])
    freq_dict = build_frequency_dict(my_bytes)
    tree = build_huffman_tree(freq_dict)
    codes = get_codes(tree)
    actual_codes = {83: '110', 84: '111', 71: '100', 72: '101', 68: '010', 69: '011', 65: '000', 67: '001'}
    assert codes == actual_codes


def test_my_avg_length() -> None:
    my_bytes = bytes([65, 65, 65, 65, 65, 65])
    freq_dict = build_frequency_dict(my_bytes)
    tree = build_huffman_tree(freq_dict)

    assert avg_length(tree, freq_dict) == 1

    my_bytes = bytes([65, 67, 65, 68, 69, 67, 68, 71, 72, 69, 71, 83, 84, 72, 83, 84])
    freq_dict = build_frequency_dict(my_bytes)
    tree = build_huffman_tree(freq_dict)

    assert avg_length(tree, freq_dict) == 3

    my_bytes = bytes([60, 61, 62, 62, 61, 63, 64, 62, 63, 64, 63, 65, 63, 64, 65, 64, 65, 64, 65, 65, 65])
    freq_dict = build_frequency_dict(my_bytes)
    tree = build_huffman_tree(freq_dict)

    assert round(avg_length(tree, freq_dict), 2) == 2.43


def test_generate_tree_postorder() -> None:
    nodes = [ReadNode(0, 104, 0, 101),
    ReadNode(0, 119, 0, 114),
    ReadNode(1, 0, 1, 1),
    ReadNode(0, 100, 0, 111),
    ReadNode(0, 108, 1, 3),
    ReadNode(1, 2, 1, 4)]
    tree = generate_tree_postorder(nodes, 5)
    number_nodes(tree)
    assert list(tree_to_bytes(tree)) == [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]

    nodes = [ReadNode(0, 3, 0, 4), ReadNode(0, 7, 1, 0), ReadNode(0, 8, 1, 1)]
    tree = generate_tree_postorder(nodes, 2)
    number_nodes(tree)
    assert list(tree_to_bytes(tree)) == [0, 3, 0, 4, 0, 7, 1, 0, 0, 8, 1, 1]

    nodes = [ReadNode(0, 4, 0, 3), ReadNode(1, 0, 0, 5), ReadNode(1, 1, 0, 6)]
    tree = generate_tree_postorder(nodes, 2)
    number_nodes(tree)
    assert list(tree_to_bytes(tree)) == [0, 4, 0, 3, 1, 0, 0, 5, 1, 1, 0, 6]

    my_bytes = bytes([65, 65, 65, 65, 65, 65])
    freq_dict = build_frequency_dict(my_bytes)
    tree = build_huffman_tree(freq_dict)

    b = tree_to_bytes(tree)
    nodes = bytes_to_nodes(b)
    tree2 = generate_tree_postorder(nodes, 0)
    assert tree == tree2

    freq_dict = {86: 8, 93: 18, 92: 2, 32: 14, 99: 67, 83: 54}
    tree = build_huffman_tree(freq_dict)
    number_nodes(tree)
    b = tree_to_bytes(tree)
    nodes = bytes_to_nodes(b)
    tree2 = generate_tree_postorder(nodes, len(nodes) - 1)
    assert tree == tree2

    freq_dict = {63: 8, 67: 6, 71: 2, 75: 4}
    tree = build_huffman_tree(freq_dict)
    number_nodes(tree)
    b = tree_to_bytes(tree)
    nodes = bytes_to_nodes(b)
    tree2 = generate_tree_postorder(nodes, len(nodes) - 1)
    assert tree == tree2

    freq_dict = {61: 100, 73: 54, 65: 72, 66: 10}
    tree = build_huffman_tree(freq_dict)
    number_nodes(tree)
    b = tree_to_bytes(tree)
    nodes = bytes_to_nodes(b)
    tree2 = generate_tree_postorder(nodes, len(nodes) - 1)
    assert tree == tree2

    zero = HuffmanTree(None, HuffmanTree(8), HuffmanTree(9))
    one = HuffmanTree(None, HuffmanTree(10), HuffmanTree(11))
    two = HuffmanTree(None, zero, one)
    three = HuffmanTree(None, HuffmanTree(6), HuffmanTree(7))
    four = HuffmanTree(None, two, three)
    five = HuffmanTree(None, HuffmanTree(12), HuffmanTree(13))
    six = HuffmanTree(None, HuffmanTree(14), HuffmanTree(15))
    seven = HuffmanTree(None, five, six)
    eight = HuffmanTree(None, HuffmanTree(16), HuffmanTree(17))
    nine = HuffmanTree(None, HuffmanTree(18), HuffmanTree(19))
    ten = HuffmanTree(None, eight, nine)
    eleven = HuffmanTree(None, seven, ten)
    twelve = HuffmanTree(None, four, eleven)
    number_nodes(twelve)
    b = tree_to_bytes(twelve)
    nodes = bytes_to_nodes(b)
    tree2 = generate_tree_postorder(nodes, len(nodes) - 1)
    assert twelve == tree2






def test_improve_tree() -> None:

    left = HuffmanTree(None, HuffmanTree(86), HuffmanTree(None, HuffmanTree(93), HuffmanTree(92)))
    right = HuffmanTree(None, HuffmanTree(None, HuffmanTree(32), HuffmanTree(99)), HuffmanTree(83))
    freq_dict = {86: 8, 93: 18, 92: 2, 32: 14, 99: 67, 83: 54}
    tree = HuffmanTree(None, left, right)
    assert round(avg_length(tree, freq_dict), 4) == 2.6196

    # optimal tree input (tree should remain the same)
    left = HuffmanTree(None, HuffmanTree(63), HuffmanTree(67))
    right = HuffmanTree(None, HuffmanTree(71), HuffmanTree(75))
    tree = HuffmanTree(None, left, right)
    freq_dict = {63: 8, 67: 6, 71: 2, 75: 4}

    old_avg = avg_length(tree)
    # original_leaves = [tree.symbol for tree in _collect_elements(tree) if tree.is_leaf()]
    improve_tree(tree, freq_dict)
    new_avg = avg_length(tree)
    # new_leaves = [tree.symbol for tree in _collect_elements(tree) if tree.is_leaf()]
    # assert original_leaves == new_leaves
    assert old_avg == new_avg

    left = HuffmanTree(61)
    right = HuffmanTree(None, HuffmanTree(73), HuffmanTree(None, HuffmanTree(65), HuffmanTree(66)))
    tree = HuffmanTree(None, left, right)
    freq_dict = {61: 100, 73: 54, 65: 72, 66: 10}
    assert round(avg_length(tree, freq_dict), 4) == 1.9237
    improve_tree(tree, freq_dict)
    assert round(avg_length(tree, freq_dict), 4) == 1.8475

    left = HuffmanTree(61)
    right = HuffmanTree(None, HuffmanTree(73), HuffmanTree(None, HuffmanTree(65), HuffmanTree(66)))
    tree = HuffmanTree(None, left, right)
    freq_dict = {61: 97, 73: 98, 65: 99, 66: 100}

    assert round(avg_length(tree, freq_dict), 4) == 2.2589
    improve_tree(tree, freq_dict)
    assert round(avg_length(tree, freq_dict), 4) == 2.2411

    right_right = HuffmanTree(None, HuffmanTree(4), HuffmanTree(5))
    right_left = HuffmanTree(3)
    right = HuffmanTree(None, right_left, right_right)
    left = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    tree = HuffmanTree(None, left, right)
    freq = {1: 1, 2: 1, 3: 1, 4: 2, 5: 1}
    improve_tree(tree, freq)
    assert round(avg_length(tree, freq), 4) == 2.3333

    left = HuffmanTree(None, HuffmanTree(99, None, None),
                            HuffmanTree(100, None, None))
    right = HuffmanTree(None, HuffmanTree(101, None, None),
                             HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    tree = HuffmanTree(None, left, right)
    freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    assert avg_length(tree, freq) == 2.49
    improve_tree(tree, freq)
    assert avg_length(tree, freq) == 2.31



if __name__ == "__main__":
    pytest.main(["test_huffman_properties_basic.py"])

