import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.ioff()
plt.style.use('fivethirtyeight')


def beta_pdf(alpha, beta):
    """ Produce curve of beta distribution.

    :param alpha: Alpha parameter of beta distribution
    :param beta: Beta parameter of beta distribution
    :return: curve of beta distribution
    """
    x = np.linspace(0, 1, 1000)
    return x, beta_dist(1 + alpha, 1 + beta).pdf(x)


def regret_visualization(random_method, thomson_method, e_greedy_method, upper_confidence_bounce_sampling,
                         thomson_method_discover, thomson_method_uncertainty):
    """ Visualize regrets of different sampling methods

    :param thomson_method_uncertainty:
    :param random_method: list of cumulative regret of random sampling
    :param thomson_method: list of cumulative regret of THomson sampling
    :param e_greedy_method: list of cumulative regret of e-greedy sampling
    :param upper_confidence_bounce_sampling: list of cumulative regret of UCB sampling
    :param thomson_method_discover: list of cumulative regret of Customized Thomson sampling
    :return:
    """
    y_label = list(range(0, len(random_method)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_label, y=random_method,
                             mode='lines',
                             name='Random Method'))
    fig.add_trace(go.Scatter(x=y_label, y=thomson_method,
                             mode='lines',
                             name='Thomson Sampling'))
    fig.add_trace(go.Scatter(x=y_label, y=e_greedy_method,
                             mode='lines',
                             name='e-greedy Method'))
    fig.add_trace(go.Scatter(x=y_label, y=upper_confidence_bounce_sampling,
                             mode='lines',
                             name='UCB'))
    fig.add_trace(go.Scatter(x=y_label, y=thomson_method_discover,
                             mode='lines',
                             name='TS discover'))
    fig.add_trace(go.Scatter(x=y_label, y=thomson_method_uncertainty,
                             mode='lines',
                             name='TS uncertainty'))
    fig.update_layout(
        title="Cumulative Regret",
        xaxis_title="Round",
        yaxis_title="Total Regret")


    fig.show()


def prob_visualization(random_method, thomson_method, e_greedy_method, upper_confidence_bounce_sampling,
                       thomson_method_discover, thomson_method_uncertainty):

    slot_names = ['# 1 slot (0.1)', '# 2 slot (0.4)', '# 3 slot (0.45)', '# 4 slot (0.6)', '# 5 slot (0.61)',]
    fig = go.Figure(data=[
        go.Bar(name='Random Method', x=slot_names, y=random_method.success_prob),
        go.Bar(name='Thomson Sampling', x=slot_names, y=thomson_method.success_prob),
        go.Bar(name='e-greedy Method', x=slot_names, y=e_greedy_method.success_prob),
        go.Bar(name='UCB', x=slot_names, y=upper_confidence_bounce_sampling.success_prob),
        go.Bar(name='TS discover', x=slot_names, y=thomson_method_discover.success_prob),
        go.Bar(name='TS uncertainty', x=slot_names, y=thomson_method_uncertainty.success_prob)
    ])

    fig.update_layout(
        barmode='group',
        title="Success Rate Pedictions by Method",
        xaxis_title="Slots",
        yaxis_title="Success Rate")

    fig.show()