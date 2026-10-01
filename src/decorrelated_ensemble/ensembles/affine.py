class AffineAggregate:
    """Two scalar aggregation parameters fitted on training OOF predictions."""

    def __init__(self, base, parameters):
        self.base, self.parameters = base, parameters

    def predict(self, X):
        return self.parameters["slope"] * self.base.predict(X) + self.parameters["intercept"]

    def capacity(self):
        return {**self.base.capacity(), "aggregation_scalar_parameters": 2}
