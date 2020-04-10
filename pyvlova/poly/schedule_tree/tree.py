from __future__ import annotations

from typing import Any, Mapping, Optional

import isl
import yaml

from pyvlova.poly.schedule_tree.node import Node, NodeWithSingleChild, \
    BandNode, DomainNode, ExtensionNode, SequenceNode, FilterNode, MarkNode


class ScheduleTree(object):
    def __init__(self, root: Optional[Node] = None):
        if isinstance(root, str):
            root = type(self).from_yaml(root).root
        self.root = root

    def copy(self) -> ScheduleTree:
        return type(self).from_yaml(self.to_yaml())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ScheduleTree:
        return cls(Node.from_dict(d))

    @classmethod
    def from_yaml(cls, s) -> ScheduleTree:
        if not isinstance(s, str):
            s = str(s)
        return cls.from_dict(yaml.load(s, Loader=yaml.Loader))

    @classmethod
    def from_file(cls, f) -> ScheduleTree:
        if isinstance(f, str):
            with open(f) as f:
                return cls.from_yaml(f.read())
        return cls.from_yaml(f.read())

    @classmethod
    def from_isl(cls, s) -> ScheduleTree:
        if isinstance(s, isl.schedule_node):
            s = s.schedule()
        return cls(Node.from_isl(s.root()))

    def to_yaml(self) -> str:
        return self.root.to_yaml()

    def to_isl(self) -> isl.schedule:
        return isl.schedule(self.to_yaml())

    def domain(self) -> isl.union_set:
        if isinstance(self.root, DomainNode):
            return self.root.domain
        elif isinstance(self.root, ExtensionNode):
            return self.root.extension.range()
        else:
            assert False

    def outermost_band(self) -> Optional[BandNode]:
        node = self.root
        while node and isinstance(node, NodeWithSingleChild) and not isinstance(node, BandNode):
            node = node.child
        if isinstance(node, BandNode):
            return node
        return None

    def outermost_band_box(self):
        return self.outermost_band().schedule_box(self.domain())

    def parallel_tilable(self):
        band = self.outermost_band()
        if band is None \
                or any(map(lambda x: not x, band.coincident)) \
                or (len(band.coincident) != 1 and not band.permutable):
            return False
        return True

    def apply_params(self, *args, **kwargs):
        return self.root.apply_params(*args, **kwargs)

    def gpu_tile(self, tile_size, permutation=None):
        assert self.parallel_tilable()
        box_size, lowers, strides = self.outermost_band_box()
        n = len(box_size)
        assert len(tile_size) == n

        real_tile_size = [tile_size[i] * strides[i] for i in range(n)]
        filled_box_size = [-(-box_size[i] // (real_tile_size[i])) * real_tile_size[i] for i in range(n)]

        thread_fake_args = ['i%d' % i for i in range(n)]
        thread_fake_constraints = [
            f'({i} mod {stride}) = (({lower}) mod {stride})'
            f' and 0 <= {i} - ({lower}) < {size}'
            for i, lower, stride, size in
            zip(thread_fake_args, lowers, strides, filled_box_size)
        ]
        thread_fake_named_tuple = f'_thread[{", ".join(thread_fake_args)}]'
        thread_fake_statement = isl.union_set(
            f'{{ {thread_fake_named_tuple} : {" and ".join(thread_fake_constraints)} }}'
        )

        old_domain = self.domain()
        if isinstance(self.root, DomainNode):
            self.root.domain = self.root.domain.union(thread_fake_statement)
        elif isinstance(self.root, ExtensionNode):
            self.root.extension = self.root.extension.union(
                isl.union_map.from_range(thread_fake_statement))
        else:
            assert False

        band = self.outermost_band()

        for i in range(n):
            s = band.schedule.at(i).union_add(
                isl.pw_aff(f'{{ {thread_fake_named_tuple} -> [({thread_fake_args[i]})] }}'))
            band.schedule = band.schedule.set_at(i, s.coalesce())

        thread_fake_branch = SequenceNode()
        thread_fake_branch.add_child(FilterNode(filter='{%s}' % thread_fake_named_tuple))
        old_branch = thread_fake_branch.add_child(FilterNode(filter=old_domain))
        if band.child:
            old_branch.child = band.child
        band.child = thread_fake_branch

        if permutation is not None:
            band.permute(*permutation)

        band.tile(*real_tile_size)

        band.insert_before(MarkNode('bind=blockIdx'))
        child = band.child
        child.insert_before(MarkNode('bind=threadIdx'))
        kernel = child.child
        kernel.insert_before(MarkNode('clear(bind)'))


"""
example_tree = ScheduleTree.from_yaml('''
domain: "[n, m, q] -> { S0[i, j]: 0 <= i < n and 0 <= j < m; S1[i, j, k]: 0 <= i < n and 0 <= j < m and 0 <= k < q}"
child:
  schedule: "[{S0[i, j] -> [(i * 2)]; S1[i, j, k] -> [(i * 2)]}, {S0[i, j] -> [(j)]; S1[i, j, k] -> [(j)]}]"
  permutable: 1
  coincident: [ 1, 1 ]
  child:
    sequence:
    - filter: '{S0[i, j]}'
    - filter: '{S1[i, j, k]}'
      child:
        schedule: "[{S1[i, j, k] -> [(k)]}]"
''')

tree = example_tree.copy()
tree.apply_params(n=1024, m=2048, q=4096)
tree.gpu_tile([391, 1096])
print(tree.to_yaml())
"""
