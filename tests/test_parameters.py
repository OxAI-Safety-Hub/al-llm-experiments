from al_llm.parameters import Parameters


def test_default_parameters_are_dummy():

    # Create a parameters instance with all the defaults
    parameters = Parameters()

    # Make sure we use dummy values
    assert "Dummy" in parameters["classifier"]
    assert "Dummy" in parameters["acquisition_function"]
    assert "Dummy" in parameters["sample_generator"]
