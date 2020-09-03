import numpy as np


class DecisionMethods:

    # initializing
    def __init__(self, epsilon=0.05):
        """ Initialize class

        :param epsilon: epsilon parameter of e-greedy algorithm. Which is threshold of random choice
        """
        self.epsilon = epsilon

    @staticmethod
    def thomson_sampling(k_array, reward_array, number_slots):
        """ Function which apply thomson sampling algorithm and return which slot machine must be played.

        :param k_array: Matrix which consist of every turn and which slot played on
        :param reward_array: Matrix which consist of every reward of every turn
        :param number_slots: How many slots we have
        :return: Id of slot machine which is chosen by algorithm
        """
        # list of samples, for each slot machine
        samples_list = []

        # successes and failures
        success_count = reward_array.sum(axis=1)
        failure_count = k_array.sum(axis=1) - success_count

        # drawing a sample from each slot distribution
        samples_list = [np.random.beta(1 + success_count[bandit_id], 1 + failure_count[bandit_id]) for bandit_id in
                        range(number_slots)]

        # returning bandit with best sample
        return np.argmax(samples_list)

    @staticmethod
    def random_sampling(k_array, reward_array, number_slots):
        """ Randomly choose one of slot machine

        :param k_array: Matrix which consist of every turn and which slot played on
        :param reward_array: Matrix which consist of every reward of every turn
        :param number_slots: How many slots we have
        :return: Id of slot machine which is chosen by algorithm
        """
        return np.random.choice(range(number_slots), 1)[0]

    def e_greedy_sampling(self, k_array, reward_array, number_slots):
        """ Function which apply e-greedy sampling algorithm and return which slot machine must be played.

        :param k_array: Matrix which consist of every turn and which slot played on
        :param reward_array: Matrix which consist of every reward of every turn
        :param number_slots: How many slots we have
        :return: Id of slot machine which is chosen by algorithm
        """

        # successes and total draws
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)

        # ratio of successes vs total
        success_ratio = success_count / total_count

        # choosing best greedy action or random depending with epsilon probability
        if np.random.random() < self.epsilon:

            # returning random action, excluding best
            return np.random.choice(np.delete(list(range(number_slots)), np.argmax(success_ratio)))

        # else return best
        else:

            # returning best greedy action
            return np.argmax(success_ratio)

    @staticmethod
    def upper_confidence_bounce_sampling(k_array, reward_array, number_slots):
        """ Function which apply Upper Confidence Bounce sampling algorithm and return which slot machine must be played.

        :param k_array: Matrix which consist of every turn and which slot played on
        :param reward_array: Matrix which consist of every reward of every turn
        :param number_slots: How many slots we have
        :return: Id of slot machine which is chosen by algorithm
        """

        # successes and total draws
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)

        # ratio of successes vs total
        success_ratio = success_count / total_count

        # computing square root term
        sqrt_term = np.sqrt(2 * np.log(np.sum(total_count)) / total_count)

        # returning best greedy action
        return np.argmax(success_ratio + sqrt_term)

    def customized_thomson_sampling_discover(self, k_array, reward_array, number_slots):
        """ Function which apply customized Thomson Sampling algorithm and return which slot machine must be played.

        :param k_array: Matrix which consist of every turn and which slot played on
        :param reward_array: Matrix which consist of every reward of every turn
        :param number_slots: How many slots we have
        :return: Id of slot machine which is chosen by algorithm
        """

        # list of samples, for each slot
        samples_list = []

        # successes and failures
        success_count = reward_array.sum(axis=1)
        failure_count = k_array.sum(axis=1) - success_count
        total_count = k_array.sum(axis=1)

        # ratio of successes vs total
        success_ratio = success_count / total_count

        # drawing a sample from each bandit distribution
        samples_list = [np.random.beta(1 + success_count[bandit_id], 1 + failure_count[bandit_id]) for bandit_id in
                        range(number_slots)]
        if np.random.random() < self.epsilon:

            # returning random action, excluding best
            return np.random.choice(np.delete(list(range(number_slots)), np.argmax(success_ratio)))
        # else return best
        else:
            # returning bandit with best sample
            return np.argmax(samples_list)

    @staticmethod
    def customized_thomson_sampling_uncertanity(k_array, reward_array, number_slots):
        """ Function which apply customized thomson sampling  algorithm and return which slot machine must be played.

        :param k_array: Matrix which consist of every turn and which slot played on
        :param reward_array: Matrix which consist of every reward of every turn
        :param number_slots: How many slots we have
        :return: Id of slot machine which is chosen by algorithm
        """
        # list of samples, for each slot machine
        samples_list = []
        prob_list = []

        # successes and failures
        success_count = reward_array.sum(axis=1)
        failure_count = k_array.sum(axis=1) - success_count

        # drawing a sample from each slot distribution
        # samples_list = [np.random.beta(1 + success_count[bandit_id], 1 + failure_count[bandit_id]) for bandit_id in
        #                 range(number_slots)]


        for slot_id in range(number_slots):
            for i in range(10):
                prob_list.append(np.random.beta(1 + success_count[slot_id], 1 + failure_count[slot_id]))
            samples_list.append(np.mean(prob_list))

        # returning bandit with best sample
        return np.argmax(samples_list)
