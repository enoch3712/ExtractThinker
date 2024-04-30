class Cascade:
    def __init__(self, models, evaluator):
        # List of models (ModelDecorator instances) in the order they should be tried
        self.models = models
        # The evaluator instance (Eval class) to use for evaluating model output
        self.evaluator = evaluator

    def process(self, input_data):
        for model in self.models:
            # Assume model.generate(input_data) returns {'content': [{'logprob': value}]}
            result = model.generate(input_data)
            if self.evaluator.evaluate(result):
                # If the evaluator passes the output, return the response
                return result.choices[0].message.content
        # If none of the models produce a satisfactory response, return an indication of failure
        return "No satisfactory response found."