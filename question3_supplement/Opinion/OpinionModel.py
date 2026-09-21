import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalVonNeumannGrid

from .OpinionAgent import OpinionAgent

class OpinionWorld(Model):
    "Model class for the Opinion World model"

    EXTREME_TOLERANCE = 0.9

    def __init__(
            self,
            lattice_side_length = 21,
            initialisation_type = "uniform",
            neighbourhood_size = None,
            learning_rate = 0.4,
            confidence_threshold = 2,
            stubborn_proportion = 0.2,
            seed=None,
        ):
        if not (learning_rate > 0) or (learning_rate > 0.5):
            raise ValueError("Learning rate must be in the interval (0, 0.5]")
        
        super().__init__(seed=seed)

        # Model parameters
        self.num_agents = lattice_side_length**2

        # Intialise environment, single capacity, toroidal boundaries
        self.grid = OrthogonalVonNeumannGrid(
            dimensions=(lattice_side_length, lattice_side_length),
            torus=True,
            random=self.random
            )

        stubborn = [False] * self.num_agents    # initially all agents aren't stubborn

        # set agent initial opinion distributions based on type
        match initialisation_type:
            case "uniform":
                opinions = self.rng.uniform(low=-1.0, high=1.0, size=self.num_agents)
            case "normal":
                opinions = self.rng.normal(loc=0, scale=0.5, size=self.num_agents)
                # ensure opinions only lie between 1 and -1
                for i in range(len(opinions)):
                    if opinions[i] < -1:
                        opinions[i] = -1
                    if opinions[i] > 1:
                        opinions[i] = 1
            case "stubborn":
                # stubborn agents begin with an unifomly distributed initial opinion
                opinions = self.rng.uniform(low=-1.0, high=1.0, size=self.num_agents)
                stubborn = []
                
                # a proportion of agents will become stubborn
                # only the agents with a negative view on the cat are eligable to be stubborn
                for i in range(len(opinions)):
                    if 0 > opinions[i] > -1 and self.rng.random() < stubborn_proportion*2:
                        stubborn.append(True)
                    else:
                        stubborn.append(False)
                        
            case _:
                raise ValueError(
                    f"unknown value of initialisation type: {initialisation_type}"
                )
        
        # if no neighbourhood size is given, assume agents are fully connected
        if neighbourhood_size is None:
            neighbourhood_size = lattice_side_length

        # Initialise agents
        OpinionAgent.create_agents(
            self,
            self.num_agents,
            self.grid.all_cells.cells,
            opinions,
            learning_rate,
            neighbourhood_size,
            confidence_threshold,
            stubborn    # initialise agents with stubborn property
            )

        # Create data collectors
        self.datacollector = DataCollector(
            model_reporters={"opinions_mean": lambda m: np.mean([a.opinion for a in m.agents]),
                           "opinions_std": lambda m: np.std([a.opinion for a in m.agents]),
                           "proportion_of_extremists": lambda m: sum([1 for a in m.agents if m.EXTREME_TOLERANCE < abs(a.opinion)]) / m.num_agents
                           },
            agent_reporters={"opinion": lambda a: a.opinion
                             },
        )
        self.datacollector.collect(self)

    def step(self):
        """
        Run one step of the model.
        """
        # All the agents chat
        self.agents.shuffle_do("chat")

        # Collect data
        self.datacollector.collect(self)
