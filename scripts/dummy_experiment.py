from al_llm import Experiment, Parameters, ProjectOption

parameters = Parameters()
dummy_args = Experiment.make_experiment(parameters, ProjectOption.Sandbox, "dummy")
experiment = Experiment(**dummy_args)
experiment.run()
