def label_and_get_results(
    samples: list,
    existing_labels: list,
    existing_ambiguities: list,
    show_feedback: bool,
) -> dict:
    """Prompts the user and compares answers against existing labels

    Parameters
    ----------
    samples : list
        A list of the samples to be labelled.
    existing_labels : list
        A list of the existing labels for these samples.
    existing_ambiguities : list
        A list of the existing ambiguities for these samples.
    show_feedback : bool
        True if the user should be told at each label whether they were
        correct or not.

    Returns
    ----------
    restults : dict
        A dictionary containing the results of the labelling process.
    """

    pass