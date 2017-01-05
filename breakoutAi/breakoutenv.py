from pybrain.rl.environments.environment import Environment
from scipy import zeros




class BreakoutEnv(Environment):
    """A break out game implementation of an environment. """

    #the number of action values the environment accepts
    indim = 3

    #the number of sensor values the environment produces
    outdim = 688

    def __init__(self,game):
        self.game = game

    def getSensors(self):
        """ the currently visible state of the world (the
            observation may be stochastic - repeated calls returning different values)
            :rtype: by default, this is assumed to be a numpy array of doubles
        """
        return [self.game.getPositions(),]


    def performAction(self,action):
        """ perform an action on the world that change it's internal state (maybe stochastically).
            :key action: an action that should be excuted in the Environment.
            :type action: by default, this is assumed to be a numpy array of doubles
        """
        self.game.comInput(action)

    def reset(self):
        """Most environments will implement this optional method that allows for reinitialization"""
