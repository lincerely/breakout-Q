from scipy import clip, asarray

from pybrain.rl.environments.task import Task
from numpy import *

class BreakoutTask(Task):
    """A task is associating a purpose with an environment.
    It decides how to evaluate the observations, potentially returning reinforcement rewards or fitness values.
    Furthermore it is a filter for what should be visible to the agent.
    Also, it can potentially act as a filter on how actions are transmitted to the environment. """

    def __init__(self,environment):
        """All tasks are coupled to an environment."""
        self.env = environment
        self.lastreward=0

    def performAction(self, action):
        """ A filtered mapping to getSample of the underlying environment. """
        self.env.performAction(action)

    def getObservation(self):
        """A filtered mapping to getSample of the underlying environment. """
        sensors = self.env.getSensors()
        return sensors

    def getReward(self):
        """Compute and return the current reward(i.e. corrsppnding to the last action performed)"""
        reward = self.env.game.getReward()
        #retrieve last reward, and save current given reward
        cur_reward = self.lastreward
        self.lastreward = reward

        return cur_reward

    @property
    def indim(self):
        return self.env.indim

    @property
    def outdim(self):
        return self.env.outdim
