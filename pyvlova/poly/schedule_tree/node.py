# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
from typing import Dict, List, Any, Type, Mapping, Optional, Iterable, Callable

import isl

from pyvlova.utils import get_unnamed_tuples


def to_isl_style_yaml(d, indent=''):
    lines = []
    for k, v in d.items():
        if isinstance(v, str):
            lines.append(f'{indent}{k}: "{v}"')
        elif isinstance(v, int):
            lines.append(f'{indent}{k}: {v}')
        elif isinstance(v, dict):
            field = to_isl_style_yaml(v, indent + ' ' * 2)
            lines.append(f'{indent}{k}:\n{field}')
        elif isinstance(v, list):
            if isinstance(next(iter(v), 0), int):
                field = ', '.join(map(str, v))
                lines.append(f'{indent}{k}: [ {field} ]')
            else:
                lines.append(f'{indent}{k}:')
                for i in v:
                    field = to_isl_style_yaml(i, indent + ' ' * 2)
                    lines.append(f'{indent}- {field.strip()}')
        else:
            assert False
    return '\n'.join(lines)


def base_dict_from_isl_node_band(node: isl.schedule_node_band) -> Dict[str, Any]:
    schedule = node.partial_schedule()
    permutable = node.permutable()
    coincident = [node.member_get_coincident(i) for i in range(schedule.size())]
    return {'schedule': schedule, 'coincident': coincident, 'permutable': permutable}


def base_dict_from_isl_node(node: isl.schedule_node) -> Dict[str, Any]:
    for t in NodeTypes:
        if t in (DomainNode, BandNode):
            continue
        if t.match_fields(dir(node)):
            return {k: getattr(node, k)() for k in t.fields}
    assert False


base_dict_from_isl: Dict[Type, Callable] = {
    isl.schedule_node_sequence: lambda x: dict(),
    isl.schedule_node_set: lambda x: dict(),
    isl.schedule_node_mark: lambda x: {'mark': x.id()},
    isl.schedule_node_domain: lambda x: {'domain': x.domain()},
    isl.schedule_node_filter: base_dict_from_isl_node,
    isl.schedule_node_extension: base_dict_from_isl_node,
    isl.schedule_node_context: base_dict_from_isl_node,
    isl.schedule_node_guard: base_dict_from_isl_node,
    isl.schedule_node_expansion: base_dict_from_isl_node,
    isl.schedule_node_band: base_dict_from_isl_node_band,
}


class Node(object):
    fields: List[str] = []
    child_repr: str = 'child'

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Optional[Node]:
        if not d:
            return None
        for t in NodeTypes:
            if t.match_fields(d.keys()):
                d = dict(d.items())
                if t.child_repr in d:
                    if issubclass(t, NodeWithSingleChild):
                        d['children'] = cls.from_dict(d.pop(t.child_repr))
                    else:
                        assert isinstance(d[t.child_repr], Iterable)
                        d['children'] = list(map(cls.from_dict, d.pop(t.child_repr)))
                return t(**d)
        assert False

    @classmethod
    def _from_isl(cls, node: isl.schedule_node) -> Optional[Node]:
        if isinstance(node, isl.schedule):
            node = node.root()
        if isinstance(node, isl.schedule_node_leaf):
            return None
        _, my_node_type = type(node).__name__.rsplit('_', 1)
        my_node_type = f'{my_node_type.capitalize()}Node'
        my_node_type = globals()[my_node_type]
        kwargs = base_dict_from_isl[type(node)](node)
        if node.n_children():
            kwargs['children'] = [cls._from_isl(node.child(i)) for i in range(node.n_children())]
            kwargs['children'] = list(filter(lambda x: x is not None, kwargs['children']))
        return my_node_type(**kwargs)

    @classmethod
    def from_isl(cls, node: isl.schedule_node) -> Optional[Node]:
        if isinstance(node, isl.schedule):
            node = node.root()
        if isinstance(node, isl.schedule_node_leaf):
            return None
        isl_root = node.root()
        root = cls._from_isl(isl_root)
        path = []
        cur = node
        while cur.has_parent():
            path.append(cur.child_position())
            cur = cur.parent()
        for i in reversed(path):
            root = root.get_child(i)
        return root

    @classmethod
    def match_fields(cls, fields: Iterable[str]):
        if cls.fields:
            return bool(set(cls.fields).intersection(fields))
        return cls.child_repr in set(fields)

    def __init__(self, parent: Optional[Node] = None, children: Iterable[Node] = None):
        self.parent: Optional[Node] = parent
        if children is None:
            children = list()
        self.children: List[Node] = list(children)
        for child in self.children:
            child.parent = self

    def get_child(self, i):
        return self.children[i]

    def set_child(self, i, node):
        self.children[i] = node
        node.parent = self

    def add_child(self, node):
        self.children.append(node)
        node.parent = self
        return node

    def child_pos_of_parent(self) -> int:
        if self.parent is None:
            return -1
        for i, v in enumerate(self.parent.children):
            if v is self:
                return i
        assert False

    def get_path_from_root(self) -> List[int]:
        if self.parent is None:
            return list()
        res = self.parent.get_path_from_root()
        res.append(self.child_pos_of_parent())
        return res

    def _children_to_vanilla(self):
        return [i.to_dict() for i in self.children]

    def to_dict(self) -> Dict[str, Any]:
        res = dict()
        for field in self.fields:
            res[field] = str(getattr(self, field))
        if self.children:
            res[self.child_repr] = self._children_to_vanilla()
        return res

    def to_yaml(self) -> str:
        # Due to isl implementation, we have to implement a customized serializer
        # return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        return to_isl_style_yaml(self.to_dict())

    def to_isl(self) -> isl.schedule_node:
        if self.parent is None:
            return isl.schedule(self.to_yaml()).get_root()
        node = self.get_root().to_isl()
        for i in self.get_path_from_root():
            node = node.child(i)
        return node

    def get_root(self) -> Node:
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def call_on_fields(self, func_name, *args, _fields=None, **kwargs):
        if _fields is None:
            _fields = self.fields
        for k in _fields:
            v = getattr(self, k)
            if hasattr(v, func_name):
                setattr(self, k, getattr(v, func_name)(*args, **kwargs))

    def all_call_on_fields(self, func_name, *args, **kwargs):
        self.call_on_fields(func_name, *args, **kwargs)
        for c in self.children:
            c.all_call_on_fields(func_name, *args, **kwargs)

    def apply_params(self, *args, **kwargs):
        if args:
            kwargs = args[0]
        for v in kwargs.values():
            assert str(v).isdigit() or str(v).isidentifier()
        c = ' and '.join((f'{k} = ({v})' for k, v in kwargs.items()))
        k = ', '.join(kwargs.keys())
        s = isl.set('[%s] -> {: %s}' % (k, c))
        self.all_call_on_fields('intersect_params', s)
        self.all_call_on_fields('project_out_all_params')

    def insert_before(self, node: Node):
        i = self.child_pos_of_parent()
        p = self.parent
        self.parent = node
        node.children = [self]
        node.parent = p
        node.parent.set_child(i, node)


class SequenceNode(Node):
    child_repr: str = 'sequence'


class SetNode(Node):
    child_repr: str = 'set'


class NodeWithSingleChild(Node):
    def __init__(self, **kwargs):
        if 'children' in kwargs:
            if isinstance(kwargs['children'], Node):
                kwargs['children'] = [kwargs['children']]
        else:
            kwargs['children'] = list()
        if 'child' in kwargs:
            assert not kwargs['children']
            child = kwargs.pop('child')
            if child:
                kwargs['children'] = [child]
        super().__init__(**kwargs)

    @property
    def child(self):
        if self.children:
            return self.get_child(0)
        return None

    @child.setter
    def child(self, child):
        if self.child is None:
            self.add_child(child)
        else:
            self.set_child(0, child)

    def _children_to_vanilla(self):
        if self.child:
            return self.child.to_dict()
        return None


class MarkNode(NodeWithSingleChild):
    fields: List[str] = ['mark']

    def __init__(self, mark=None, **kwargs):
        super().__init__(**kwargs)
        if mark is None:
            mark = ''
        self.mark: str = str(mark)


class DomainNode(NodeWithSingleChild):
    fields: List[str] = ['domain']

    def __init__(self, domain=None, **kwargs):
        super().__init__(**kwargs)
        if domain is None or isinstance(domain, str):
            domain = isl.union_set(domain or '{}')
        self.domain: isl.union_set = domain


class FilterNode(NodeWithSingleChild):
    fields: List[str] = ['filter']

    def __init__(self, **kwargs):
        if 'filter' in kwargs:
            filter_ = kwargs.pop('filter')
        else:
            filter_ = None
        super().__init__(**kwargs)
        if filter_ is None or isinstance(filter_, str):
            filter_ = isl.union_set(filter_ or '{}')
        self.filter: isl.union_set = filter_


class ExtensionNode(NodeWithSingleChild):
    fields: List[str] = ['extension']

    def __init__(self, extension=None, **kwargs):
        super().__init__(**kwargs)
        if extension is None or isinstance(extension, str):
            extension = isl.union_map(extension or '{}')
        self.extension: isl.union_map = extension


class ContextNode(NodeWithSingleChild):
    fields: List[str] = ['context']

    def __init__(self, context=None, **kwargs):
        super().__init__(**kwargs)
        if context is None or isinstance(context, str):
            context = isl.set(context or '{ : }')
        elif isinstance(context, isl.union_set):
            if context.is_empty():
                context = isl.set('{ : }')
            else:
                assert context.isa_set()
                context = isl.set.from_union_set(context)
        self.context: isl.set = context


class GuardNode(NodeWithSingleChild):
    fields: List[str] = ['guard']

    def __init__(self, guard=None, **kwargs):
        super().__init__(**kwargs)
        if guard is None or isinstance(guard, str):
            guard = isl.set(guard or '{ : }')
        elif isinstance(guard, isl.union_set):
            if guard.is_empty():
                guard = isl.set('{ : }')
            else:
                assert guard.isa_set()
                guard = isl.set.from_union_set(guard)
        self.guard: isl.set = guard


class ExpansionNode(NodeWithSingleChild):
    fields: List[str] = ['expansion', 'contraction']

    def __init__(self, expansion=None, contraction=None, **kwargs):
        super().__init__(**kwargs)
        if expansion is None or isinstance(expansion, str):
            expansion = isl.union_map(expansion or '{}')
        if contraction is None or isinstance(contraction, str):
            contraction = isl.union_pw_multi_aff(contraction or '{}')
        self.expansion: isl.union_map = expansion
        self.contraction: isl.union_pw_multi_aff = contraction


class BandNode(NodeWithSingleChild):
    fields: List[str] = ['schedule', 'coincident', 'permutable']

    def __init__(self, schedule=None, coincident=None, permutable=None, **kwargs):
        super().__init__(**kwargs)
        if schedule is None or isinstance(schedule, str):
            schedule = isl.multi_union_pw_aff(schedule or '[]')
        if coincident is None:
            coincident = [False] * schedule.size()
        else:
            coincident = list(map(bool, coincident))
        self.schedule: isl.multi_union_pw_aff = schedule
        self.coincident: List[bool] = coincident
        self.permutable: bool = bool(permutable)

    def call_on_fields(self, func_name, *args, **kwargs):
        if func_name == 'intersect_params':
            return
        return super().call_on_fields(func_name, *args, **kwargs)

    def normalize(self) -> BandNode:
        if len(self.coincident) < self.schedule.size():
            self.coincident.extend(
                [False] * (self.schedule.size() - len(self.coincident)))
        elif len(self.coincident) > self.schedule.size():
            self.coincident = self.coincident[:self.schedule.size()]
        return self

    def to_dict(self) -> Dict[str, Any]:
        node = copy.copy(self).normalize()
        res = dict()
        res['schedule'] = str(node.schedule)
        res['coincident'] = list(map(int, map(bool, node.coincident)))
        res['permutable'] = int(bool(node.permutable))
        if self.child:
            res[self.child_repr] = self._children_to_vanilla()
        return res

    def to_isl(self) -> isl.schedule_node_band:
        res = super().to_isl()
        assert isinstance(res, isl.schedule_node_band)
        return res

    def tile(self, *tile_size):
        if len(tile_size) == 1 and isinstance(tile_size[0], Iterable):
            tile_size = tuple(tile_size[0])
        t = isl.multi_val('{[%s]}' % ', '.join(map(str, tile_size)))
        node = self.from_isl(self.to_isl().tile(t))
        self.child.insert_before(node.child)
        for i in self.fields:
            setattr(self, i, getattr(node, i))
        return self

    def permute(self, *permutation):
        n = self.schedule.size()
        if n <= 1:
            return
        if len(permutation) == 1 and isinstance(permutation[0], Iterable):
            permutation = tuple(permutation[0])
        assert frozenset(permutation) == frozenset(range(n))
        old_schedule = [self.schedule.at(i) for i in range(n)]
        for i in range(n):
            self.schedule = self.schedule.set_at(i, old_schedule[permutation[i]])

    def schedule_box(self, domain):
        n = self.schedule.size()
        band_map = isl.union_map.convert_from(self.schedule)
        s = domain.apply(band_map).coalesce()
        assert s.isa_set()
        s = isl.set.from_union_set(s)
        box = s.simple_fixed_box_hull()
        strides = [int(str(s.stride(i))) for i in range(n)]
        band_size, *_ = get_unnamed_tuples(box.size())
        band_size = list(map(int, band_size))
        lowers, *_ = get_unnamed_tuples(isl.point.from_pw_multi_aff(box.offset()))
        return band_size, lowers, strides


NodeTypes: List[Type[Node]] = [
    SequenceNode, SetNode, MarkNode, DomainNode, FilterNode,
    ExtensionNode, ContextNode, GuardNode, ExpansionNode, BandNode
]
