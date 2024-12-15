import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict

class Node:
  def __init__(self, state, parent = None, action = None, g=0, h=0, f=0):
    self.state = state
    self.parent = parent
    self.action = action
    self.g = g
    self.h = h
    self.f = f


class GenericAgent():

    def search(self, env: CampusEnv, *args, **kwargs) -> Tuple[List[int], float, int]:
        init_state = env.get_initial_state()
        result = self._search(env, *args, **kwargs)
        env.set_state(init_state)
        return result


class DFSGAgent(GenericAgent):
    def __init__(self) -> None:
        self.env = None
        self.expanded = 0

    def solution(self, node, pathe, cost)-> Tuple[List[int], int]:
        if node.parent is None:
          return pathe, cost
        self.env.set_state(node.parent.state)
        state, trans_cost , terminated  = self.env.step(node.action)
        pathe.insert(0,node.action)
        return self.solution(node.parent, pathe, cost + trans_cost)

    def dfs_search_helper(self , open_list, visited) -> Tuple[List[int], int, int]:
        current_node = open_list.pop()
        # self.expanded = 0
        visited.add(current_node.state)
        if self.env.is_final_state(current_node.state):
            actions, cost = self.solution(current_node,[],0)
            return actions, cost, self.expanded
        self.expanded+=1
        if None in self.env.succ(current_node.state)[0]:   #current_node is a hole
            return ([], -1, -1)
        for action, successor in self.env.succ(current_node.state).items():
            self.env.set_state(current_node.state)
            child = Node(successor[0], current_node, action)
            states_list = [s.state for s in open_list]
            if child.state in visited or child.state in states_list:
               continue
            else:
                open_list.append(child)
                result = self.dfs_search_helper(open_list, visited)
                if bool(result[0]):
                   return result
        return [], -1, -1

    def _search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.expanded = 0
        init_node = Node(self.env.get_initial_state())
        open_list=[init_node]
        visited=set()
        return self.dfs_search_helper(open_list,visited)


class UCSAgent(GenericAgent):

    def __init__(self) -> None:
        self.env = None

    def solution(self, node, actions)-> Tuple[List[int], int]:
        if node.parent is None:
            return actions, 0
        self.env.set_state(node.parent.state)
        actions.insert(0, node.action)
        return self.solution(node.parent, actions)

    def _search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.env=env
        expanded=0
        open_dict=heapdict.heapdict()
        init_state = Node(self.env.get_initial_state())
        open_dict[init_state]=(0,0) # (cost, state)
        visited = set()
        while bool(open_dict):
            c_node_key=open_dict.popitem()
            c_node=c_node_key[0]
            # print({1})
            visited.add(c_node.state)
            if self.env.is_final_state(c_node.state):
                # print({2})
                actions, x =self.solution(c_node,[])
                cost=c_node.g
                return actions, cost, expanded
            expanded+=1
            for action, successor in self.env.succ(c_node.state).items():
                if successor[1] is None:
                    continue
                child =Node(successor[0], c_node, action,c_node.g+successor[1])
                states_list=[s[1] for s in open_dict.values()]
                if child.state not in states_list and child.state not in visited :
                    open_dict[child]=(child.g,child.state)
                elif child.state in states_list :
                    to_delete=[]
                    for key,val in open_dict.items():
                        if val[0] > child.g and val[1] == child.state:
                            to_delete.append(key)
                    if bool(to_delete):
                        open_dict[child]=(child.g,child.state)
                        del open_dict[to_delete[0]]
        return [], -1, -1


class WeightedAStarAgent(GenericAgent):

    def __init__(self):
        self.env = None
        self.expanded = 0

    # ℎ𝑀𝑎𝑛ℎ𝑎𝑡𝑎𝑛(𝑠, 𝑔) g in graph
    def h_calculator_aux(self, state):
        x, y = self.env.to_row_col(state)
        g_state_list = []
        g_state_list = self.env.get_goal_states()
        g_list = [self.env.to_row_col(v) for v in g_state_list]
        return min(abs(x-v[0]) + abs(y-v[1]) for v in g_list)

    # heuristic calculator
    def h_calculator(self, state):
        return min(100,self.h_calculator_aux(state))

    # f value calculator
    def c_f_val(self, h, g , h_weight):
        new_as_score = (h_weight*h) + ((1-h_weight)*g)
        return new_as_score

    # t_cost== the total cost
    # node is the cureent state
    def solution(self, node, actions, t_cost)-> Tuple[List[int], int]:
        if node.parent is None:
            return actions,t_cost
        self.env.set_state(node.parent.state)
        state, c_cost, finshed=self.env.step(node.action)
        actions.insert(0, node.action)
        return self.solution(node.parent, actions, c_cost+t_cost)

    def _search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env=env
        open_dict=heapdict.heapdict()
        self.expanded = 0
        init_heuristic=self.h_calculator(self.env.get_initial_state())
        init_f=self.c_f_val(init_heuristic, 0, h_weight)
        init_state = Node(self.env.get_initial_state(),h=self.h_calculator(self.env.get_initial_state()),f=init_f)
        # print({init_f})
        open_dict[init_state]=(init_f,0)    # ( score , state)
        visited=set()
        while bool(open_dict):
            c_node_key=open_dict.popitem()
            c_node=c_node_key[0]
            visited.add(c_node)
            # print({c_node.state})
            #
            #
            #
            #
            if self.env.is_final_state(c_node.state):
                actions,cost = self.solution(c_node,[],0)
                return actions, cost, self.expanded
            self.expanded += 1
            # graph.succ Returns the successors of the state.
            for action, successor in self.env.succ(c_node.state).items():
                if successor[1] is None:
                    continue
                next_g=c_node.g+successor[1]
                next_h=self.h_calculator(successor[0])
                next_f=self.c_f_val(next_h,next_g,h_weight)                          #h_weight
                # if self.expanded in [1, 2]:
                #     print(self.env.succ(c_node.state))
                #     print(f"n_state {successor[0]}")
                #     print(f"expand {self.expanded}")
                #     print(f" next_g {next_g}")
                # indicates how much weight is given to

                next_node=Node(successor[0],c_node,action,next_g,next_h,next_f)
                # if next_node.state not in [i[1] for i in open_dict]:
                found_in_visited = False
                for node in visited:
                    if node.state == next_node.state:
                        found_in_visited = True

                if next_node.state not in [i[1] for i in open_dict.values()]:
                    # print(f"not in open: {next_node.state}")
                    # for element in visited:
                        # print(element.state )
                    # if next_node. not in visited:
                    if not found_in_visited:
                        # print(f"not in visited: {next_node.state}")
                        open_dict[next_node]=(next_f,next_node.state) # ( score , state)
                        # for key, value in open_dict.items():
                        #     print(f": {value}")
                        continue
                    # node we open it before we visted
                    else:
                        # print("AAA")
                        for node in visited:
                            if node.state == next_node.state:
                                # print(f" node_s {node.state}")
                                # print(f" next_f {next_f}")
                                if next_f < node.f:
                                    # print(f" node_s {node.state}")
                                    # print(f" next_f {next_f}")
                                    # print(f" node_f {node.f}")
                                    open_dict[next_node]=(next_f,next_node.state)
                                    visited.remove(node)
                                    break
                # if the node in the open dict
                # elif next_node not in visited:
                elif not found_in_visited:
                    # print("BBBB")
                    key_found = None
                    for key,val in open_dict.items():
                        if val[1] == next_node.state:
                            key_found = key
                            break
                    # n_key=open_dict[next_node][0]
                    if next_f < key_found.f:
                       open_dict.pop(key_found)
                       open_dict[next_node]=(next_f,next_node.state)

        return [],0,0


class AStarAgent(WeightedAStarAgent):

    def _search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        return super()._search(env, 0.5)
