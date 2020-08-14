# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from pyvlova.poly import ScheduleTree, Statement, TensorTable


example_schedule_tree_yaml = '''
domain: "[n, m, q] -> { S0[i, j]: 0 <= i < n and 0 <= j < m; S1[i, j, k]: 0 <= i < n and 0 <= j < m and 0 <= k < q}"
child:
  schedule: "[{S0[i, j] -> [(i)]; S1[i, j, k] -> [(i)]}, {S0[i, j] -> [(j)]; S1[i, j, k] -> [(j)]}]"
  permutable: 1
  coincident: [ 1, 1 ]
  child:
    sequence:
    - filter: '{S0[i, j]}'
    - filter: '{S1[i, j, k]}'
      child:
        schedule: "[{S1[i, j, k] -> [(k)]}]"
'''
example_tree = ScheduleTree.from_yaml(example_schedule_tree_yaml)

def S0(tensor_table, i, j):
    tensor_table['C'][i, j] = 0.0

def S1(tensor_table, i, j, k):
    tensor_table['C'][i, j] = tensor_table['C'][i, j] + tensor_table['A'][i, k] * tensor_table['B'][k, j]

def get_example_tensor_table(n, m, q):
    t = TensorTable()
    t.add_tensor('A', [n, q])
    t.add_tensor('B', [q, m])
    t.add_tensor('C', [n, m])
    return t

example_stmt_table = {'S0': Statement.from_calc(S0), 'S1': Statement.from_calc(S1)}
