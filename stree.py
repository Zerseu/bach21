import re
import sys


class SuffixTreeNode:
    new_identifier = 0

    def __init__(self, start=0, end=sys.maxsize):
        self.identifier = SuffixTreeNode.new_identifier
        SuffixTreeNode.new_identifier += 1
        self.suffix_link = None
        self.edges = {}
        self.parent = None
        self.bit_vector = 0
        self.start = start
        self.end = end

    def add_child(self, key, start, end):
        child = SuffixTreeNode(start=start, end=end)
        child.parent = self
        self.edges[key] = child
        return child

    def add_existing_node_as_child(self, key, node):
        node.parent = self
        self.edges[key] = node

    def get_edge_length(self, current_index):
        return min(self.end, current_index + 1) - self.start

    def __str__(self):
        return 'id=' + str(self.identifier)


class SuffixTree:
    def __init__(self):
        self.root = SuffixTreeNode()
        self.input_string = ''
        self.strings_count = 0
        self.leaves = []

    def append_string(self, input_string):
        start_index = len(self.input_string)
        current_string_index = self.strings_count
        input_string += '$' + str(current_string_index)
        self.input_string += input_string
        self.strings_count += 1
        active_node = self.root
        active_edge = 0
        active_length = 0
        remainder = 0
        new_leaves = []
        for index in range(start_index, len(self.input_string)):
            previous_node = None
            remainder += 1
            while remainder > 0:
                if active_length == 0:
                    active_edge = index
                if self.input_string[active_edge] not in active_node.edges:
                    leaf_node = active_node.add_child(self.input_string[active_edge], index, sys.maxsize)
                    leaf_node.bit_vector = 1 << current_string_index
                    new_leaves.append(leaf_node)
                    if previous_node is not None:
                        previous_node.suffix_link = active_node
                    previous_node = active_node
                else:
                    next_node = active_node.edges[self.input_string[active_edge]]
                    next_edge_length = next_node.get_edge_length(index)
                    if active_length >= next_node.get_edge_length(index):
                        active_edge += next_edge_length
                        active_length -= next_edge_length
                        active_node = next_node
                        continue
                    if self.input_string[next_node.start + active_length] == self.input_string[index]:
                        active_length += 1
                        if previous_node is not None:
                            previous_node.suffix_link = active_node
                        previous_node = active_node
                        break
                    split_node = active_node.add_child(
                        self.input_string[active_edge],
                        next_node.start,
                        next_node.start + active_length
                    )
                    next_node.start += active_length
                    split_node.add_existing_node_as_child(self.input_string[next_node.start], next_node)
                    leaf_node = split_node.add_child(self.input_string[index], index, sys.maxsize)
                    leaf_node.bit_vector = 1 << current_string_index
                    new_leaves.append(leaf_node)
                    if previous_node is not None:
                        previous_node.suffix_link = split_node
                    previous_node = split_node
                remainder -= 1
                if active_node == self.root and active_length > 0:
                    active_length -= 1
                    active_edge = index - remainder + 1
                else:
                    active_node = active_node.suffix_link if active_node.suffix_link is not None else self.root
        for leaf in new_leaves:
            leaf.end = len(self.input_string)
        self.leaves.extend(new_leaves)

    def find_longest_common_substrings(self):
        success_bit_vector = 2 ** self.strings_count - 1
        lowest_common_ancestors = []
        for leaf in self.leaves:
            node = leaf
            while node.parent is not None:
                if node.bit_vector != success_bit_vector:
                    node.parent.bit_vector |= node.bit_vector
                    node = node.parent
                else:
                    lowest_common_ancestors.append(node)
                    break
        longest_common_substrings = ['']
        longest_length = 0
        for common_ancestor in lowest_common_ancestors:
            common_substring = ''
            node = common_ancestor
            while node.parent is not None:
                label = self.input_string[node.start:node.end]
                common_substring = label + common_substring
                node = node.parent
            common_substring = re.sub(r'(.*?)\$?\d*$', r'\1', common_substring)
            if len(common_substring) > longest_length:
                longest_length = len(common_substring)
                longest_common_substrings = [common_substring]
            elif len(common_substring) == longest_length and common_substring not in longest_common_substrings:
                longest_common_substrings.append(common_substring)
        return longest_common_substrings


def main():
    suffix_tree = SuffixTree()
    suffix_tree.append_string('AnaBanana')
    suffix_tree.append_string('BananaAna')
    lcs = suffix_tree.find_longest_common_substrings()
    for s in lcs:
        print(s)


if __name__ == '__main__':
    main()
