"""Adaptive controller stub for hardware loops."""

class AdaptiveController:
    def __init__(self, agent=None):
        self.agent = agent

    def update(self, metric, value):
        """Dummy update method."""
        if self.agent:
            self.agent.publish(metric, value)
