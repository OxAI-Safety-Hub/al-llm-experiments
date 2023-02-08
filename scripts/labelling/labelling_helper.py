import textwrap
from typing import OrderedDict, Tuple


def _wrap(text: str) -> str:
    """Wrap some text to the line width"""
    return textwrap.fill(text, width=70)


def _prompt(
    sample: str,
    categories: OrderedDict,
    prompt_ambiguities: bool,
) -> Tuple[int, int]:
    """Prompts the user for the label and ambiguity of a sentence

    Parameters
    ----------
    sample : str
        The sample to label.
    categories : OrderedDict
        The human readable categories to define samples by.
    prompt_ambiguities : bool
        True if the human should also provide ambiguities

    Returns
    ----------
    new_label, new_ambiguity : int, int
        A tuple of the label and ambiguity defined by the user.
    """

    # Build the message with the sample plus the category selection
    text = _wrap(f"{sample!r}") + "\n"
    text += _wrap("How would you classify this?") + "\n"
    for j, cat_human_readable in enumerate(categories.values()):
        text += _wrap(f"[{j}] {cat_human_readable}") + "\n"
    # If also checking for ambiguity, add these options
    if prompt_ambiguities:
        for j, cat_human_readable in enumerate(categories.values()):
            text += (
                _wrap(f"[{j+len(categories)}] {cat_human_readable} (ambiguous)") + "\n"
            )

    # Print the message
    print(text)

    # Keep asking the user for a label until they give a valid one
    if not prompt_ambiguities:
        max_valid_label = len(categories) - 1
    else:
        max_valid_label = 2 * len(categories) - 1
    prompt = _wrap(f"Enter a number (0-{max_valid_label}):")
    valid_label = False
    while not valid_label:
        label_str = input(prompt)
        try:
            label = int(label_str)
        except ValueError:
            continue
        if label >= 0 and label <= max_valid_label:
            valid_label = True

    new_label = label % len(categories)
    new_ambiguity = label // len(categories)
    return new_label, new_ambiguity


def label_and_get_results(
    samples: list,
    existing_labels: list,
    existing_ambiguities: list,
    categories: OrderedDict,
    show_feedback: bool,
    score_ambiguities: bool,
) -> Tuple[list, list, dict]:
    """Prompts the user and compares answers against existing labels

    Parameters
    ----------
    samples : list
        A list of the samples to be labelled.
    existing_labels : list
        A list of the existing labels for these samples.
    existing_ambiguities : list
        A list of the existing ambiguities for these samples.
    categories : OrderedDict
        The human readable categories to define samples by.
    show_feedback : bool
        True if the user should be told at each label whether they were
        correct or not.
    score_ambiguities : bool
        True if ambiguities should have to match as well as the labels.

    Returns
    ----------
    new_labels : list
        A list of the new labels.
    new_ambiguities : list
        A list of the new ambiguities.
    restults : dict
        A dictionary containing the results of the labelling process.
    """

    # list to store labelled values
    new_labels = []
    new_ambiguities = []

    num_consistent_labels = 0

    # for each sample that needs labelling
    for i in range(len(samples)):
        # prompt for label and ambiguity
        print(f"\nSample [{i+1}/{len(samples)}]")
        label, ambiguity = _prompt(samples[i], categories, score_ambiguities)

        # record these values
        new_labels.append(label)
        new_ambiguities.append(ambiguity)

        # does this new label match the existing one
        labels_match = new_labels[i] == existing_labels[i]
        if score_ambiguities:
            ambiguities_match = new_ambiguities[i] == existing_ambiguities[i]
            labels_match = labels_match and ambiguities_match

        # if the labels match, update consistency
        if labels_match:
            num_consistent_labels += 1

        # show live feedback on results
        if show_feedback:
            if labels_match:
                print("This was the correct label!")
            else:
                print("This label was incorrect.")
                if not score_ambiguities:
                    print(f"It should have been: {existing_labels[i]}")
                else:
                    print(
                        f"It should have been: Label: {existing_labels[i]}, Ambiguity: {existing_ambiguities[i]}"
                    )

    # assemble and return results
    results = {"consistency": num_consistent_labels / len(samples)}
    return new_labels, new_ambiguities, results
