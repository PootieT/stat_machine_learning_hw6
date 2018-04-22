# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(iterations):
          temp = self.values.copy()
          for state in self.mdp.getStates():
            print "in main loop iteration:",i,", calculating state:",state
            bestA = self.computeActionFromValues(state)
            if bestA is not None:
              temp[state] = self.computeQValueFromValues(state, bestA)
          self.values = temp

        

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        q_val = 0
        for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
          print "from ",state," to ",nextState,"transition probability:",prob,",reward:", self.mdp.getReward(state,action,nextState),",value:", self.discount*self.getValue(nextState)
          q_val += prob*(self.mdp.getReward(state,action,nextState) + self.discount*self.getValue(nextState))
        return q_val


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if self.mdp.isTerminal(state):
          print "found terminal state"
          return None
        else:
          # maxQ = -float("inf")
          # bestA = self.mdp.getPossibleActions(state)[0]
          # for action in self.mdp.getPossibleActions(state):
          #   q_val = self.computeQValueFromValues(state, action)
          #   print "evaluating action:",action,", Q-value:",q_val
          #   if q_val > maxQ:
          #     maxQ = q_val
          #     bestA = action
          # print "best action:",bestA
          # return bestA
          a_q_counter = util.Counter()
          for action in self.mdp.getPossibleActions(state):
          	a_q_counter[action] = self.computeQValueFromValues(state, action)
      	  return a_q_counter.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
