from al_llm import Experiment, Parameters

parameters = Parameters()
dummy_args = Experiment.make_experiment(parameters, "dummy")
experiment = Experiment(**dummy_args)
experiment.run_full()
