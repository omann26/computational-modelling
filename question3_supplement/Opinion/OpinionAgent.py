from mesa.discrete_space import CellAgent

class OpinionAgent(CellAgent):
    "Agent in the Opinion World."

    def __init__(
            self, 
            model, 
            cell,
            opinion,
            learning_rate,
            neighbourhood_size,
            confidence_threshold,
            stubborn    # add a stubborn property
        ):
        super().__init__(model)

        self.cell = cell
        self.opinion = opinion
        self.learning_rate = learning_rate
        self.neighbourhood_size = neighbourhood_size
        self.confidence_threshold = confidence_threshold
        self.stubborn = stubborn

    def update_opinion(self, other):
        if self.stubborn == False:  # opinions only change if the agent isn't stubborn
            if abs(self.opinion - other.opinion) <= self.confidence_threshold:
                # apply positive influence   
                self.opinion += self.learning_rate * (other.opinion - self.opinion)

    def chat(self):
        # Pick a neighbour to chat with
        chat_partner = self.cell.get_neighborhood(radius=self.neighbourhood_size).select_random_agent()

        # Both update opinions
        self.update_opinion(chat_partner)
        chat_partner.update_opinion(self)
