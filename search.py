# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from os import popen
from util import *
from game import Directions
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def convDir(dir):
    if dir == "North":
        return Directions.NORTH
    if dir == "South":
        return Directions.SOUTH
    if dir == "East":
        return Directions.EAST
    if dir == "West":
        return Directions.WEST
    print(f"{dir} is not accepted!")
    return None


def depthFirstSearch(problem):
    
    frontier = Stack()
    parents = {}
    parents[problem.getStartState()] = (None, None)
    frontier.push(problem.getStartState())
    expanded = []
    node = None
    while not frontier.isEmpty():
        node = frontier.pop()
        # print(f"Popped {node}")
        if problem.isGoalState(node):
            nodePath = []
            actions = []
            pnode = node
            act = None
            while pnode != None:
                nodePath.append(pnode)
                actions.append(act)
                pnode, act = parents[pnode]
            nodePath.append(problem.getStartState())
            nodePath.reverse()
            actions.pop(0)
            actions.reverse()
            print(f"NodePath:{nodePath}")
            print(f"Actions:{actions}")
            return actions

        if node not in expanded:
            # print(f"Explored {node}")
            expanded.append(node)
            for child in problem.getSuccessors(node):
                if child[0] not in expanded:
                    # print(f"Pushed {child[0]}")
                    parents[child[0]] = (node, child[1])
                    frontier.push(child[0])
    return None
    
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    print(problem.getStartState())
    frontier = Queue()
    expanded = []
    parents = {}
    frontier.push(problem.getStartState())
    parents[problem.getStartState()] = (None, None)
    node = None
    while not frontier.isEmpty():
        node = frontier.pop()  
        if problem.isGoalState(node):
            nodePath = []
            actions = []
            pnode = node
            act = None #individual direction
            while pnode != None:
                nodePath.append(pnode)
                actions.append(act)
                pnode, act = parents[pnode]
            nodePath.append(problem.getStartState())
            nodePath.reverse()
            actions.pop(0)
            actions.reverse()
            return actions
        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                if child[0] not in expanded:
                    parents[child[0]] = (node, child[1])
                    print(child)
                    frontier.push(child[0])
    return None
    
    
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = PriorityQueue()
    parents = {}
    costs = {}
    parents[problem.getStartState()] = (None, None)
    frontier.push(problem.getStartState(),0)
    costs[problem.getStartState()] = 0
    expanded = []
    node = None
    while not frontier.isEmpty():
        node = frontier.pop()
        # print(f"Popped {node}")
        if problem.isGoalState(node):
            nodePath = []
            actions = []
            pnode = node
            act = None
            while pnode != None:
                nodePath.append(pnode)
                actions.append(act)
                pnode, act = parents[pnode]
            nodePath.append(problem.getStartState())
            nodePath.reverse()
            actions.pop(0)
            actions.reverse()
            # print(f"NodePath:{nodePath}")
            # print(f"Actions:{actions}")
            print(actions)
            return actions

        if node not in expanded:
            # print(f"Explored {node}")
            expanded.append(node)
            for child in problem.getSuccessors(node):
                if child[0] not in expanded:
                    # print(f"Pushed {child}")
                    if child[0] in costs:
                        if costs[child[0]] < costs[node] + child[2]:
                            continue
                        else:
                            costs[child[0]] = costs[node] + child[2]
                    else:
                        costs[child[0]] = costs[node] + child[2]
                    print(f"{child} COST:{costs[child[0]]}")
                    parents[child[0]] = (node, child[1])
                    frontier.push(child[0], costs[child[0]])
    return None
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    frontier = PriorityQueue()
    expanded = []
    parents = {}
    fn = {} #node, estimated actual cost
    gn = {} #node, actual cost to cur node
    gn[problem.getStartState()]=0
    frontier.push(problem.getStartState(), 0)
    parents[problem.getStartState()] = (None, None)
    node = None
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node):
            nodePath = []
            actions = []
            pnode = node
            act = None
            while pnode != None:
                nodePath.append(pnode)
                actions.append(act)
                pnode, act = parents[pnode]
            nodePath.append(problem.getStartState())
            nodePath.reverse()
            actions.pop(0)
            actions.reverse()
            return actions

        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                # print(f"{node}'s child is:{child}")
                if child[0] not in expanded:
                    hn = heuristic(child[0], problem)
                    if child[0] not in gn:
                        gn[child[0]] = gn[node] + child[2]
                    else:
                        if gn[child[0]] > gn[node] + child[2]:
                            gn[child[0]] = gn[node] + child[2]
                        else:
                            continue
                    fn[child[0]] = hn + gn[child[0]]
                    parents[child[0]] = (node, child[1])
                    frontier.update(child[0], fn[child[0]])

    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
