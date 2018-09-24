import numpy as np
import warnings

from .encoding import codes_to_query

from .debug import debug_print
VERBOSITY = 0


def collect_and_verify_clf_classlabels(m_list, m_codes):
    """
    Collect all the classlabels

    Parameters
    ----------
    m_list: list, shape (nb_models,)
        List of all the component models
    m_codes:
        List of all the codes of the MERCS model

    Returns
    -------

    """

    _, m_targ, _ = codes_to_query(m_codes)

    nb_atts = len(m_codes[0])
    clf_labels = initialize_classlabels(nb_atts, mode='default')

    for m_idx, m in enumerate(m_list):
        # Collect the classlabels of one model
        m_classlabels = collect_classlabels(m)

        # Verify all the classlabels
        clf_labels = update_clf_labels(clf_labels, m_classlabels, m_targ[m_idx])

    return clf_labels


def collect_classlabels(m):
    """
    Collect all the classlabels of a given model m.

    A given model m can be a composite model, or a sklearn model. There are
    four scenarios that we can encounter:
        1. No classes_ attribute.
            This happens when a sklearn model is fully numeric.
            We assume a fully numeric model.
        2. Classes_ attribute is None
            This also happens when a sklearn model is fully numeric. Sklearn is
            slightly inconsistent in this regard.
            We assume a fully numeric model
        3. m.classes_ is numpy array
            This is the output of a single-target sklearn model. We wrap this in
            a regular python array, to achieve consistency with multi-target
            models.
        4. m.classes_ is a list
            This happens in multi-target sklearn models, but also in our own
            composite models. This is the consistent form that we are looking form
            Here, we do nothing because this is what we require.

    Parameters
    ----------
    m: {sklearn, composite model}
        The model under consideration

    Returns
    -------

    """

    if not hasattr(m, 'classes_'):
        # If no classlabels are present, assume a fully numerical model
        m_classlabels = initialize_classlabels(m.n_outputs_, mode='numeric')
    elif m.classes_ is None:
        # If no classlabels are present, assume a fully numerical model
        m_classlabels = initialize_classlabels(m.n_outputs_, mode='numeric')
    elif isinstance(m.classes_, np.ndarray):
        # Single-target sklearn output; wrap in array
        m_classlabels = [m.classes_]
    elif isinstance(m.classes_, list):
        m_classlabels = m.classes_
    else:
        msg = "Did not recognize the classlabels: {} of this model: {}".format(m.classes_, m)
        raise TypeError(msg)

    return m_classlabels


def update_clf_labels(clf_labels, m_classlabels, m_targ):
    """
    Update the classlabels.

    Update the classlabels known to the MERCS system, based on
    potentially new information on classlabels from a new component model.

    Parameters
    ----------
    clf_labels: list, shape (nb_atts, (nb_classlabels_att,))
        List of all classlabels known to the MERCS system
    m_classlabels: list, shape (nb_targ_atts_mod, (nb_classlabels_att,))
        List of all the classlabels known to the individual model
    m_targ
        List of all targets of the individual model. These are essential to
        identify about which attributes m_classlabels is providing information!

    Returns
    -------

    """

    for t_idx, t in enumerate(m_targ):

        old_labels = clf_labels[t]          # Classlabels known to MERCS
        new_labels = m_classlabels[t_idx]   # Classlabels known to the model m

        msg = "New_labels are: {}\n" \
              "Type new_labels is: {}\n".format(new_labels, type(new_labels))
        debug_print(msg, V=VERBOSITY, warn=True)
        msg = "Old_labels are: {}\n" \
              "Type old_labels is: {}\n".format(old_labels, type(old_labels))
        debug_print(msg, V=VERBOSITY, warn=True)

        assert isinstance(old_labels, (list, np.ndarray))
        assert isinstance(new_labels, (list, np.ndarray))

        if isinstance(old_labels, list):
            if old_labels == initialize_classlabels(1, mode='default')[0]:
                # Replace default value
                clf_labels[t] = new_labels
            elif old_labels == initialize_classlabels(1, mode='numeric')[0]:
                # Both old and new labels must agree on being numeric
                assert new_labels == initialize_classlabels(1, mode='numeric')[0]
            else:
                msg = """
                type(old_labels): \t{} is list\n
                However, not the default value, nor the default value for a numeric
                attribute. \n
                These are the only two cases in which we expect an entry of clf_labels
                to be a list and not a np.ndarray.\n
                Something must be wrong.
                """.format(type(old_labels))
                raise TypeError(msg)
        elif isinstance(old_labels, np.ndarray):
            # Join current m_classlabels with those already present
            classlabels_list = [old_labels, new_labels]
            clf_labels[t] = join_classlabels(classlabels_list)
        else:
            msg = """
            old_labels (=clf_labels[t]) can only be a list or np.ndarray.\n
            A list can only occur in two case: \n
            \t 1) Default entry: [0] \n
            \t 2) Numeric dummy entry: ['numeric]\n\n
            """
            raise TypeError(msg)

    return clf_labels


def join_classlabels(classlabels_list):
    """
    Get the union of the provided classlabels

    This is crucial whenever models are trained on different subsets of the
    data_csv, and have other ideas about what the classlabels are.
    """

    all_unique_classes = np.unique(np.concatenate(classlabels_list))
    all_unique_classes.sort()

    return all_unique_classes


def initialize_classlabels(nb_atts, mode='default'):

    if mode in {'default'}:
        classlabels = [['default'] for i in range(nb_atts)]
    elif mode in {'numeric'}:
        classlabels = [['numeric'] for i in range(nb_atts)]
    else:
        msg = "Did not recognize mode: {}. Assuming 'default'".format(mode)
        warnings.warn(msg)
        classlabels = initialize_classlabels(nb_atts, mode='default')

    return classlabels
