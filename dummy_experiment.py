from al_llm import Experiment

dummy_args = Experiment.make_dummy_experiment()
experiment = Experiment(**dummy_args)
experiment.run()