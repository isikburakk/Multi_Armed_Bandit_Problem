import numpy as np


class MultiArmedBandit:

    # initialization
    def __init__(self):
        pass

    # function that helps us play with the slot machines
    @staticmethod
    def play(k, slot_probabilities):
        """ Simulate one play of slot machine with its owen probability of success.

        :param k: Index number of slot machine
        :param slot_probabilities: Probabilities of slot machines
        :return: the reward and the regret of the action
        """
        try:
            return np.random.binomial(1, slot_probabilities[k]), np.max(slot_probabilities) - slot_probabilities[k]
        except Exception as e:
            raise """ Pickle reader exception: {}""".format(e)