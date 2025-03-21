# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import math
from util import nearestPoint
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MyOffensiveAgent', second = 'MyDefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    # self.start = gameState.getAgentPosition(self.index)
    self.initFood = len(self.getFood(gameState).asList())
    CaptureAgent.registerInitialState(self, gameState)
    
    # print(f"Initial food = {self.init_food}")

    '''
    Your initialization code goes here, if you need any.
    '''
    
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    # print(actions)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)
    
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    # print(action)
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    raise NotImplementedError

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    raise NotImplementedError
  
class MyOffensiveAgent(DummyAgent):
  def getFeatures(self, gameState, action):
    
    
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    #left food
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)
  
    #food distance
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance_food = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance_food
    
    #new features
    
    hasFood = gameState.getAgentState(self.index).numCarrying
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
    myPos = successor.getAgentState(self.index).getPosition()
    
    
    #return feature
    teamIndex = self.getTeam(gameState)
    defenderIndex = teamIndex[teamIndex!=self.index]
    defenderState = successor.getAgentState(defenderIndex)
    defenderPos = defenderState.getPosition()
    defenderDistance = self.getMazeDistance(myPos, defenderPos)
    features['returnFood'] = hasFood * defenderDistance
    if len(ghosts) > 0:
      minDistance_ghost = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts])
      features['returnFood'] = 2 * hasFood * defenderDistance if minDistance_ghost < 5 else hasFood * defenderDistance
    
    
    #avoid ghost
    if len(ghosts) >0:
      scaredTime = ghosts[0].scaredTimer
      # print(f"scaredTime = {scaredTime}")
      minDistance_ghost = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts])
      if scaredTime < 20:
        if minDistance_ghost <2:
          features['distanceToGhost'] = 1.5 *(hasFood + 0.5) * math.exp(-0.5 * minDistance_ghost)
          
        else:
          features['distanceToGhost'] = (hasFood + 0.5) * math.exp(-0.5 * minDistance_ghost) 
          # if closer than bigger
          # features['distanceToGhost'] = (hasFood + 0.5) / (minDistance_ghost + 1) #plus 0.1 to avoid divided by 0
          # features['distanceToGhost'] = (hasFood + 0.5) / (minDistance_ghost + 1)**2
          # features['distanceToGhost'] = (math.exp(-0.5 * minDistance_ghost) + 1 / (minDistance_ghost + 1)**2)/2
        
      else:
        features['distanceToGhost'] = -0.5
    else:# no ghost at beginning
        features['distanceToGhost'] = 0
    # print(features)
    
      #capsule distance (try to eat capsule)
    features['distanceToCapsule'] = 0
    capsuleList = self.getCapsules(successor)
    if len(capsuleList) > 0: # if there is a capsule on map
      
      minDistance_cap = min([self.getMazeDistance(myPos, cap) for cap in capsuleList])
      if len(ghosts) > 0 and minDistance_ghost < 5:
        features['distanceToCapsule'] = minDistance_cap * 3
      else:
        features['distanceToCapsule'] = minDistance_cap
    
    # features['capsuleExists'] = 1 if len(capsuleList) > 0 else 0
    
    
    # encourage cross border
    # features['crossBorder'] = 0
    # if (not gameState.getAgentState(self.index).isPacman) and (hasFood ==0):
    #   nextIsEnemy = successor.getAgentState(self.index).isPacman
    #   features['crossBorder'] = -1 if nextIsEnemy else 0
      
      
    # avoid dead end
    features['deadEnd'] = 0
    if len(ghosts)>0:
      minDistance_ghost = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts])
      numAvailable = len([a for a in successor.getLegalActions(self.index) if a != Directions.STOP])
      if minDistance_ghost <5 and numAvailable <=1:
        features['deadEnd'] = 1 #need to check sign
        
    return features
  
  def getWeights(self, gameState, action):
    return {'successorScore': 1000, 'distanceToFood': -5,
            'distanceToGhost': 10, 'distanceToCapsule': -5,
            # 'capsuleExists': -100,
            'returnFood': -20,
            # 'crossBorder': 10,
            'deadEnd': -1000
            }
  
  
  
class MyDefensiveAgent(DummyAgent):
    
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    
    # Computes whether we're on defense (1) or offense (0) (check if defender stays in homeland)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see 
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
    
    #if defender is stop, punish
    if action == Directions.STOP: features['stop'] = 1
    
    #if defender goes back, punish
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    #new features
    #defend capsule 
    capsuleList = self.getCapsulesYouAreDefending(successor)
    if len(capsuleList) > 0: # if there is a capsule (ours) on map
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance_cap = min([self.getMazeDistance(myPos, cap) for cap in capsuleList])
      features['distanceToCapsule'] = minDistance_cap
    else:
      features['distanceToCapsule'] = 0
    return features
  
  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100,
            'invaderDistance': -20, 'stop': -100, 'reverse': -1,
            'distanceToCapsule': -5}