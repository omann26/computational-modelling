import numpy as np
from matplotlib.figure import Figure
import solara

from mesa.visualization import (
    SolaraViz,
    SpaceRenderer,
    Slider,
)
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.utils import update_counter

from Opinion.OpinionModel import OpinionWorld

def agent_portrayal(agent):
    return AgentPortrayalStyle(
        color=agent.opinion+1
    )

def post_process_grid(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "learning_rate": Slider(
        label="learning rate",
        value=0.4,
        min=0.05,
        max=0.5,
        step=0.05,
    ),
    "confidence_threshold": Slider(
        label="confidence threshold",
        value=2.0,
        min=0.05,
        max=2.0,
        step=0.05,
    ),
    "neighbourhood_size": Slider(
        label="neighbourhood size",
        value=21,
        min=1,
        max=21,
        step=1,
    ),
    "initialisation_type": {
        "type": "InputText",
        "value": "uniform",
        "label": "Initialisation type",
    }
}

model = OpinionWorld()

@solara.component
def OpinionHistogram(model):
    update_counter.get()  # This is required to update the counter
    # Note: you must initialize a figure using this method instead of
    # plt.figure(), for thread safety purpose
    fig = Figure()
    ax = fig.subplots()
    opinion_vals = [agent.opinion for agent in model.agents]
    # Note: you have to use Matplotlib's OOP API instead of plt.hist
    # because plt.hist is not thread-safe.
    ax.hist(opinion_vals, bins=np.linspace(-1, 1, 11))
    ax.set_ylim(top=model.num_agents)
    ax.set_title('Agent opinions', fontsize=20)
    ax.set_xlabel('opinion')
    ax.set_ylabel('number of agents')
    solara.FigureMatplotlib(fig)

renderer = SpaceRenderer(model, backend="matplotlib"
                         ).setup_agents(agent_portrayal, cmap="coolwarm", vmin=0, vmax=2)
# In this case the renderer only draws the agents because we just want to observe
# the state of the agents, not the structure of the grid.
renderer.draw_agents()
renderer.post_process = post_process_grid

page = SolaraViz(
    model,
    renderer,
    components=[(OpinionHistogram, 0)],
    model_params=model_params,
    name="OpinionModel",
)