"""
Assignment 2 starter code

"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    frequency_dict = {}
    for byte in text:
        frequency_dict[byte] = frequency_dict.get(byte, 0) + 1
    return frequency_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    freq_another = freq_dict.copy()
    if not freq_another:
        return None
    else:
        if len(freq_another) == 1:
            for val in freq_dict:
                return HuffmanTree(None, HuffmanTree(None), HuffmanTree(val))
        else:
            row = []
            frequencies = []
            for symbol, freq in freq_another.items():
                row.append(HuffmanTree(symbol))
                frequencies.append(freq)

            while len(row) != 1:
                for value in range(len(row) - 1):

                    smallest_value = frequencies.index(min(frequencies))
                    if smallest_value is not None:
                        curr_node = row.pop(smallest_value)
                        load = frequencies.pop(smallest_value)
                        smaller_value = frequencies.index(min(frequencies))
                        if smaller_value is not None:
                            next_currnode = row.pop(smaller_value)
                            load1 = frequencies.pop(smaller_value)
                            # create a new node with these two nodes as children
                            parent = HuffmanTree(None, curr_node, next_currnode)
                            weight_ = load + load1
                            row.append(parent)
                            frequencies.append(weight_)
                return row[0]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    symbol_table = {}
    stack = [(tree, "")]

    while stack:
        curr, binary = stack.pop()

        if curr is None:
            continue
        if curr.is_leaf():
            symbol_table[curr.symbol] = binary
        else:
            l_curr = curr.left
            r_curr = curr.right
            code_l = binary + "0"
            code_r = binary + "1"
            stack.append((l_curr, code_l))
            stack.append((r_curr, code_r))

    return symbol_table


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """

    def first_postorder_traversal(tree: HuffmanTree) -> None:
        if tree is None:
            return

        first_postorder_traversal(tree.left)
        first_postorder_traversal(tree.right)

        if not tree.is_leaf():
            tree.number = next(num_generator)

    num_generator = iter(range(0, 2 ** 31))
    first_postorder_traversal(tree)


next_num = 0


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """

    def weighted_len_paths(node: HuffmanTree,
                           path_len: int) -> float:
        if node is None or node.is_leaf():
            return freq_dict.get(node.symbol, 0.0) * path_len
        else:
            total_len = 0.0
            for small_curr in [node.left, node.right]:
                total_len += weighted_len_paths(small_curr,
                                                path_len + 1)
            return total_len

    if freq_dict:
        total_frequency = sum(freq_dict.values())
        return weighted_len_paths(tree, 0) / total_frequency
    else:
        return 0


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """

    list_of_b = [codes[symbol] for symbol in text]
    bits_str = ''.join(list_of_b)

    bits_str += '0' * (-len(bits_str) % 8)

    # Convert the padded bits string to bytes
    alr_compress = bytearray()
    temp = 0
    left_ov = len(bits_str) % 8
    if left_ov:
        bits_str += '0' * (8 - left_ov)
    while temp < len(bits_str):
        bits = bits_str[temp:temp + 8]
        if len(bits) == 8:
            byte = int(bits, 2)
            alr_compress.append(byte)
        temp += 8

    return bytes(alr_compress)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>. The
    representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

        Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
                # doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    temp_list = []

    def second_postorder(tree):
        """add the nodes into the list"""
        if not tree:
            return

        second_postorder(tree.left)
        second_postorder(tree.right)

        if tree.left and tree.right:
            temp_list.append(tree)

    # Call the function with the given tree
    second_postorder(tree)
    outcome = []
    for tree in temp_list:
        for node in [tree.left, tree.right]:
            if node.is_leaf():
                m = (0, node.symbol)
            else:
                m = (1, node.number)
            outcome.append(m)

    outcome2 = []
    for tup in outcome:
        for obje in tup:
            outcome2.append(obje)

    return bytes(outcome2)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """

    while True:
        for leftty, rightty, leftda, rightda in [(node_lst[root_index].l_type,
                                                  node_lst[root_index].r_type,
                                                  node_lst[root_index].l_data,
                                                  node_lst[root_index].r_data)]:
            children = []
            for temp in range(2):
                data = leftda if temp == 0 else rightda
                if (temp == 0 and leftty == 0) or (temp == 1 and rightty == 0):
                    children.append(HuffmanTree(data))
                else:
                    children.append(generate_tree_general(node_lst, data))
            node = HuffmanTree(None, *children)
            node.number = root_index
            return node


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    while not (node_lst[root_index].r_type == 0 and node_lst[
        root_index].l_type == 0):
        if not node_lst[root_index].l_type and node_lst[root_index].r_type == 1:
            leftsub = HuffmanTree(node_lst[root_index].l_data)
            rightsub = generate_tree_postorder(node_lst, (root_index - 1))
            node = HuffmanTree(None, leftsub, rightsub)
            break
        if node_lst[root_index].l_type == 1 and not node_lst[root_index].r_type:
            index = root_index - 1
            postorder_tree = generate_tree_postorder(node_lst, index)
            right_data = node_lst[root_index].r_data
            node = HuffmanTree(None, postorder_tree, HuffmanTree(right_data))
            break
        else:
            for i in range(1):
                right_subtree = generate_tree_postorder(node_lst,
                                                        (root_index - 1))
                number_nodes(right_subtree)
            if right_subtree is not None:
                new_index = root_index - 1 - (right_subtree.number + 1)
                node = HuffmanTree(None,
                                   generate_tree_postorder(node_lst, new_index),
                                   right_subtree)
            else:
                node = HuffmanTree(value=None, left=None, right=right_subtree)
            break

    while node_lst[root_index].l_type == 0 and node_lst[root_index].r_type == 0:
        left_subtree = HuffmanTree(node_lst[root_index].l_data)
        right_subtree = HuffmanTree(node_lst[root_index].r_data)
        node = HuffmanTree(None, left_subtree, right_subtree)
        break

    number_nodes(node)
    return node


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    curr_str = ''
    for byte in text:
        curr_str += byte_to_bits(byte)

    current = tree
    lst = []
    m = 0
    while len(lst) < size:
        if m >= len(curr_str):
            break

        if curr_str[m] == '0':
            current = current.left
        elif curr_str[m] == '1':
            current = current.right

        if current.symbol is not None:
            lst.append(current.symbol)
            current = tree

        m += 1

    return bytes(lst)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """

    def get_key(item):
        return (len(item), item)

    dict_freqcop = dict(freq_dict)
    code_list = sorted(get_codes(tree).values(), key=get_key)
    i = 0
    while i < len(code_list):
        current = tree
        for bit in code_list[i]:
            if bit == '0':
                current = current.left
            elif bit == '1':
                current = current.right
        freq_high = max(dict_freqcop, key=dict_freqcop.get)
        current.symbol = freq_high
        del dict_freqcop[freq_high]
        i += 1


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
