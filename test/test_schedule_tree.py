import unittest

from pyvlova.poly import ScheduleTree, cuda_tile


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


class TestScheduleTree(unittest.TestCase):
    def setUp(self):
        self.example_tree = ScheduleTree.from_yaml(example_schedule_tree_yaml)

    def test_cuda_tile(self):
        tree = self.example_tree.copy()
        tree.apply_params(n=1024, m=2048, q=4096)
        cuda_tile(tree, [391, 1096])
        result = str(tree.to_yaml())
        expect = '''
domain: "{ _block[i0, i1] : (i0) mod 391 = 0 and (i1) mod 1096 = 0 and 0 <= i0 <= 1172 and 0 <= i1 <= 2191; S0[i, j] : 0 <= i <= 1023 and 0 <= j <= 2047; S1[i, j, k] : 0 <= i <= 1023 and 0 <= j <= 2047 and 0 <= k <= 4095; _thread[i0, i1] : 0 <= i0 <= 1172 and 0 <= i1 <= 2191 }"
child:
  mark: "bind=blockIdx"
  child:
    schedule: "[{ _thread[i0, i1] -> [(i0 - (i0) mod 391)]; S1[i, j, k] -> [(i - (i) mod 391)]; _block[i0, i1] -> [(i0 - (i0) mod 391)]; S0[i, j] -> [(i - (i) mod 391)] }, { _thread[i0, i1] -> [(i1 - (i1) mod 1096)]; S1[i, j, k] -> [(j - (j) mod 1096)]; _block[i0, i1] -> [(i1 - (i1) mod 1096)]; S0[i, j] -> [(j - (j) mod 1096)] }]"
    coincident: [ 1, 1 ]
    permutable: 1
    child:
      mark: "bind=threadIdx"
      child:
        schedule: "[{ _thread[i0, i1] -> [((i0) mod 391)]; S1[i, j, k] -> [((i) mod 391)]; _block[i0, i1] -> [((i0) mod 391)]; S0[i, j] -> [((i) mod 391)] }, { _thread[i0, i1] -> [((i1) mod 1096)]; S1[i, j, k] -> [((j) mod 1096)]; _block[i0, i1] -> [((i1) mod 1096)]; S0[i, j] -> [((j) mod 1096)] }]"
        coincident: [ 1, 1 ]
        permutable: 1
        child:
          mark: "clear(bind)"
          child:
            sequence:
            - filter: "{ _thread[i0, i1] }"
            - filter: "{ _block[i0, i1] }"
            - mark: "clear(bind)"
              child:
                filter: "{ S0[i, j] : 0 <= i <= 1023 and 0 <= j <= 2047; S1[i, j, k] : 0 <= i <= 1023 and 0 <= j <= 2047 and 0 <= k <= 4095 }"
                child:
                  sequence:
                  - filter: "{ S0[i, j] }"
                  - filter: "{ S1[i, j, k] }"
                    child:
                      schedule: "[{ S1[i, j, k] -> [(k)] }]"
                      coincident: [ 0 ]
                      permutable: 0
'''
        self.assertEqual(result.strip(), expect.strip())


if __name__ == '__main__':
    unittest.main()
