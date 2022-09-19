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

def depthFirstSearch(problem):
    
    frontier = Stack()
    parents = {}
    frontier.push(problem.getStartState())
    expanded = []
    node = None
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node):
            path = []
            pnode = node
            while pnode != problem.getStartState():
                path.append(pnode)
                pnode = parents[pnode]
            path.append(problem.getStartState())
            path.reverse()
            actions = []
            for c in range(0,len(path)-1):
                dx = path[c+1][0] - path[c][0]
                dy = path[c+1][1] - path[c][1]
                if dx == 1:
                    actions.append(Directions.EAST)
                    continue
                elif dx == -1:
                    actions.append(Directions.WEST)
                    continue
                elif dy == 1:
                    actions.append(Directions.NORTH)
                    continue
                elif dy == -1:
                    actions.append(Directions.SOUTH)
                    continue
                else:
                    return None
            return actions

        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                if child[0] not in expanded:
                    parents[child[0]] = node
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
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
  
    frontier = Queue()
    expanded = []
    frontier.push(problem.getStartState())
    parents = {}
    action = []
    node = None
    while not frontier.isEmpty():
        node = frontier.pop()  
        print(node)     
        if problem.isGoalState(node):
            path=[]
            curr = node
            while curr != problem.getStartState():
                path.append(curr)
                curr=parents[curr]
            path.append(problem.getStartState())
            path.reverse()
            for c in range(0,len(path)-1):
                print(f"{path[c+1][0]} and {path[c][0]}")
                diffX = int(path[c+1][0]) - int(path[c][0])
                diffY = int(path[c+1][1]) - int(path[c][1])
                if diffX == 1:
                    action.append(Directions.EAST)
                    continue
                elif diffX == -1:
                    action.append(Directions.WEST)
                    continue
                elif diffY == 1:
                    action.append(Directions.NORTH)
                    continue
                elif diffY == -1:
                    action.append(Directions.SOUTH)
                    continue
                else:
                    return None
            return action
        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                if child[0] not in expanded:
                    parents[child[0]] = node
                    frontier.push(child[0])
    return None
    
    
    
    
    
    
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = PriorityQueue()
    frontier.push(problem.getStartState())
    expanded = Counter()
    node = None
    while not frontier.isEmpty():
        lastNode = node
        node = frontier.pop()
        # parents.put(node, lastNode)
        print(f"Popped {node}")
        if problem.isGoalState(node):
            path = []
            # curr = node
            # while curr != problem.getStartState():
            return path
        if node not in expanded:
            print(f"Explored {node}")
            expanded.append(node)
            for child in problem.getSuccessors(node):
                if child[0] not in expanded:
                    print(f"Pushed {child[0]}")
                    frontier.push(child[0])
                # path.push(child[1])
    return -1

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    frontier = PriorityQueue()
    parents = {}
    frontier.push(problem.getStartState())
    expanded = []
    node = None
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node):
            path = {}
            return None

        if node not in expanded:
            expanded.append(node)
            for child in problem.getSuccessors(node):
                if child[0] not in expanded:
                    parents[child[0]] = node
                    frontier.push(child[0])
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
