# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import defaultdict
from functools import reduce
from typing import Any, Mapping, Optional

import yaml

from ..._ext import isl

from .node import Node, NodeWithSingleChild, BandNode, DomainNode, ExtensionNode, SequenceNode, FilterNode, MarkNode


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
        self.root.apply_params(*args, **kwargs)

    def add_to_domain(self, uset):
        if isinstance(self.root, DomainNode):
            self.root.domain = self.root.domain.union(uset)
        elif isinstance(self.root, ExtensionNode):
            self.root.extension = self.root.extension.union(
                isl.union_map.from_range(uset))
        else:
            assert False
