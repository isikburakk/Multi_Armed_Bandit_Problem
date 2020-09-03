import numpy as np
from .helper import beta_pdf
from .multiarmedbandit import MultiArmedBandit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm
plt.ioff()
plt.style.use('fivethirtyeight')


class Simulation:


    def __init__(self, decision_method, number_play, slot_probabilities):
        """ Initialize class

        :param decision_method: Method object of sampling algorithm
        :param number_play: Number of turn of play
        :param slot_probabilities: Probability of winning for slot machines
        """

        self.decision_method = decision_method
        self.number_play = number_play
        self.slot_probabilities = slot_probabilities
        self.k_matrix = np.zeros((len(self.slot_probabilities), self.number_play))
        self.reward_matrix = np.zeros((len(self.slot_probabilities), self.number_play))
        self.k_list = []
        self.reward_list = []
        self.success_count = []
        self.failure_count = []
        self.regret_cum_list = []
        self.success_prob = 0

    def bayesian_simulation(self):
        """ Simulate the game and record results

        :return:
        """
        number_slots = len(self.slot_probabilities)
        regret_list = []

        for play_number in range(self.number_play):
            # record information about this draw
            k = self.decision_method(self.k_matrix, self.reward_matrix, number_slots)
            reward, regret = MultiArmedBandit.play(k, self.slot_probabilities)

            # record information about this draw
            regret_list.append(regret)
            self.regret_cum_list.append(np.sum(regret_list))
            self.k_list.append(k)
            self.reward_list.append(reward)
            self.k_matrix[k, play_number] = 1
            self.reward_matrix[k, play_number] = reward

            # sucesses and failures for our beta distribution
            self.success_count.append(self.reward_matrix.sum(axis=1))
            z = self.k_matrix.sum(axis=1) - self.success_count
            self.failure_count.append(z)

        self.success_prob = self.success_count[-1]/self.k_matrix.sum(axis=1)

    def animation(self, plot_title):
        """ Create a animation object for simulation process

        :param plot_title: Title of plot
        :return: Animation object
        """
        posterior_anim_dict = {i: [] for i in range(len(self.slot_probabilities))}
        slot_colors = ['red', 'green', 'blue', 'purple', 'orange']
        fig = plt.figure(figsize=(9, 4), dpi=100)
        # let us position our plots in a grid, the largest being our plays
        ax1 = plt.subplot2grid((7, 5), (0, 0), colspan=5, rowspan=4)
        ax2 = plt.subplot2grid((7, 5), (5, 0), rowspan=3)
        ax3 = plt.subplot2grid((7, 5), (5, 1), rowspan=3)
        ax4 = plt.subplot2grid((7, 5), (5, 2), rowspan=3)
        ax5 = plt.subplot2grid((7, 5), (5, 3), rowspan=3)
        ax6 = plt.subplot2grid((7, 5), (5, 4), rowspan=3)

        for i in range(self.number_play):
            for slot_id in range(len(self.slot_probabilities)):
                x, curve = beta_pdf(self.success_count[i][slot_id], self.failure_count[i][0][slot_id])
                posterior_anim_dict[slot_id].append({'X': x, 'curve': curve})

            # getting list of colors that tells us the bandit
            color_list = [slot_colors[k] for k in self.k_list]

            # getting list of facecolors that tells us the reward
            facecolor_list = [['none', slot_colors[self.k_list[i]]][r] for i, r in enumerate(self.reward_list)]
        # fixing properties of the plots
        ax1.set(xlim=(-1, self.number_play), ylim=(-0.5, len(self.slot_probabilities) - 0.5))
        ax1.set_title(plot_title, fontsize=10)
        ax1.set_xlabel('Round', fontsize=10)
        ax1.set_ylabel('Slot', fontsize=10)
        ax1.set_yticks([0, 1, 2, 3, 4])
        ax1.set_yticklabels(['{}\n($\\theta = {}$)'.format(i, self.slot_probabilities[i]) for i in range(5)])
        ax1.tick_params(labelsize=10)

        # titles of distribution plots
        ax2.set_title('Estimated $\\theta_0$', fontsize=10);
        ax3.set_title('Estimated $\\theta_1$', fontsize=10);
        ax4.set_title('Estimated $\\theta_2$', fontsize=10);
        ax5.set_title('Estimated $\\theta_3$', fontsize=10);
        ax6.set_title('Estimated $\\theta_4$', fontsize=10);

        # initializing with first data
        scatter = ax1.scatter(y=[self.k_list[0]], x=[list(range(self.number_play))[0]], color=[color_list[0]], linestyle='-',
                              marker='o', s=30, facecolor=[facecolor_list[0]]);
        dens1 = ax2.fill_between(posterior_anim_dict[0][0]['X'], 0, posterior_anim_dict[0][0]['curve'], color='red',
                                 alpha=0.7)
        dens2 = ax3.fill_between(posterior_anim_dict[1][0]['X'], 0, posterior_anim_dict[1][0]['curve'], color='green',
                                 alpha=0.7)
        dens3 = ax4.fill_between(posterior_anim_dict[2][0]['X'], 0, posterior_anim_dict[2][0]['curve'], color='blue',
                                 alpha=0.7)
        dens4 = ax5.fill_between(posterior_anim_dict[3][0]['X'], 0, posterior_anim_dict[3][0]['curve'], color='purple',
                                 alpha=0.7)
        dens5 = ax6.fill_between(posterior_anim_dict[4][0]['X'], 0, posterior_anim_dict[4][0]['curve'], color='yellow',
                                 alpha=0.7)

        def animate(i):
            """ Animation function which will be called at FucnAnimation function

            :param i: Number of frame
            :return:
            """

            # clearing axes
            ax1.clear();
            ax2.clear();
            ax3.clear();
            ax4.clear();
            ax5.clear();

            # updating game rounds
            scatter = ax1.scatter(y=self.k_list[:i], x=list(range(self.number_play))[:i], color=color_list[:i],
                                  linestyle='-', marker='o', s=30, facecolor=facecolor_list[:i]);

            # fixing properties of the plot
            ax1.set(xlim=(-1, self.number_play), ylim=(-0.5, len(self.slot_probabilities) - 0.5))
            ax1.set_title(plot_title, fontsize=10)
            ax1.set_xlabel('Round', fontsize=10);
            ax1.set_ylabel('Slots', fontsize=10)
            ax1.set_yticks([0, 1, 2, 3, 4])
            ax1.set_yticklabels(['{}\n($\\theta = {}$)'.format(i, self.slot_probabilities[i]) for i in range(5)])
            ax1.tick_params(labelsize=10)

            # updating distributions
            dens1 = ax2.fill_between(posterior_anim_dict[0][i]['X'], 0, posterior_anim_dict[0][i]['curve'], color='red',
                                     alpha=0.7)
            dens2 = ax3.fill_between(posterior_anim_dict[1][i]['X'], 0, posterior_anim_dict[1][i]['curve'],
                                     color='green', alpha=0.7)
            dens3 = ax4.fill_between(posterior_anim_dict[2][i]['X'], 0, posterior_anim_dict[2][i]['curve'],
                                     color='blue', alpha=0.7)
            dens4 = ax5.fill_between(posterior_anim_dict[3][i]['X'], 0, posterior_anim_dict[3][i]['curve'],
                                     color='purple', alpha=0.7)
            dens5 = ax6.fill_between(posterior_anim_dict[4][i]['X'], 0, posterior_anim_dict[4][i]['curve'],
                                     color='orange', alpha=0.7)

            # titles of distribution plots
            ax2.set_title('Estimated $\\theta_0$', fontsize=10);
            ax3.set_title('Estimated $\\theta_1$', fontsize=10);
            ax4.set_title('Estimated $\\theta_2$', fontsize=10);
            ax5.set_title('Estimated $\\theta_3$', fontsize=10);
            ax5.set_title('Estimated $\\theta_4$', fontsize=10);

            # do not need to return
            return ()

        # function for creating animation
        anim = FuncAnimation(fig, animate, frames=self.number_play, interval=100, blit=True)

        # fixing the layout
        fig.tight_layout()

        return anim
