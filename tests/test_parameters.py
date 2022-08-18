from al_llm.parameters import Parameters


def test_default_parameters_are_dummy():

    # Create a parameters instance with all the defaults
    parameters = Parameters()

    # Make sure we use dummy values
    assert parameters["classifier"] == "dummy"
    assert parameters["acquisition_function"] == "dummy"
    assert parameters["sample_generator_base_model"] == "dummy"
