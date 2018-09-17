import numpy as np
import pandas as pd
import warnings


# Everything related to codes
def codes_to_query(codes, atts=None):
    """
    Change the codes-array to an actual queries, which are three arrays.

    :param code:                Array that contains: 0-desc/1-target/-1-missing code for each attribute
    :param atts:                Array that contains the attributes (indices)
    :return: Three arrays.      One for desc atts indices, one for targets, one for missing
    """

    if atts is None: atts = list(range(len(codes[0])))
    nb_codes = len(codes)

    desc, targ, miss = [], [], []

    for c_idx in range(nb_codes):
        c_desc, c_targ, c_miss = code_to_query(codes[c_idx], atts)

        desc.append(c_desc)
        targ.append(c_targ)
        miss.append(c_miss)

    return desc, targ, miss


def code_to_query(code, atts=None):
    """
    Change the code-array to an actual queries, which are three arrays.

    :param code:                Array that contains:
                                     0 for desc attribute
                                     1 for target attribute
                                    -1 for missing attribute
    :param atts:                Array that contains the attributes (indices)
    :return: Three arrays.      One for desc atts indices, one for targets,
                                one for missing

    TODO(elia): The coding strategy is still hardcoded here. Fix this.
    """

    if atts is None: atts = list(range(len(code)))
    assert len(code) == len(atts)

    desc = [x for i, x in enumerate(atts)
            if code[i] == encode_attribute(x,[x],[])]
    targ = [x for i, x in enumerate(atts)
            if code[i] == encode_attribute(x,[],[x])]
    miss = [x for i, x in enumerate(atts)
            if code[i] == encode_attribute(x, [], [])]
    return desc, targ, miss


def query_to_code(q_desc, q_targ, q_miss, atts=None):
    if atts is None:
        atts = determine_atts(q_desc, q_targ, q_miss)

    code = [encode_attribute(a, q_desc, q_targ) for a in atts]

    return code


def queries_to_codes(q_desc, q_targ, q_miss, atts=None):
    assert len(q_desc) == len(q_targ) == len(q_miss)
    nb_queries = len(q_desc)

    if atts is None:
        atts = determine_atts(q_desc[0], q_targ[0], q_miss[0])

    codes = [query_to_code(q_desc[i], q_targ[i], q_miss[i], atts=atts)
             for i in range(nb_queries)]

    return codes


def determine_atts(desc, targ, miss):
    """
    Determine the entire list of attributes.
    """
    atts = list(set(desc + targ + miss))
    atts.sort()
    return atts


def encode_attribute(att, desc, targ):
    """
    Encode the 'role' of an attribute in a model.

    `Role` means:
        - Descriptive attribute (input)
        - Target attribute (output)
        - Missing attribute (not relevant to the model)
    """

    check_desc = att in desc
    check_targ = att in targ

    code_int = check_targ * 2 + check_desc - 1

    return code_int


# Related to settings['metadata']
## Actual metadata
def get_metadata_df(df):
    """
    Get some useful statistics from a Pandas DataFrame.

    :param df:  Input DataFrame
    :return:    Dict with metadata
    #TODO(elia) is_nominal should become is_type
    """

    nb_tuples = df.shape[0]
    nb_atts = df.shape[1]

    # Initialize arrays (-1/False are no arbitrary choices!)
    nb_uvalues = np.full(nb_atts, -1)
    is_nominal = np.full(nb_atts, 0)
    tr_nominal = 20 # Our threshold of nb classes to call a column 'nominal'

    # Fill when necessary
    for i, col in enumerate(df.columns):
        if pd.api.types.is_integer_dtype(df[col]):
            nb_uvalues[i] = df[col].nunique()
            is_nominal[i] = 1 if nb_uvalues[i] < tr_nominal else 0

    metadata = {'is_nominal':   is_nominal,
                'nb_atts':      nb_atts,
                'nb_tuples':    nb_tuples,
                'nb_values':    nb_uvalues}

    return metadata


## Classlabels
def collect_and_verify_clf_classlabels(m_list, m_codes, is_nominal = None):
    """
    Collect the labels of the classifier.

    :param m_codes:
    :param m_list:
    :return:
    """

    _, m_targ, _ = codes_to_query(m_codes)

    nb_atts =  len(m_codes[0])
    clf_labels = [[0]] * nb_atts # Fill with dummy classes

    for m_idx, m in enumerate(m_list):
        # Collect the classlabels of one model
        nb_targ = len(m_targ[m_idx])
        m_classlabels = collect_classlabels(m, nb_targ, m_idx=m_idx)

        # Verify all the classlabels
        clf_labels = update_clf_labels(clf_labels, m_classlabels, m_targ[m_idx])

    return clf_labels


def collect_classlabels(m, nb_targ, m_idx=0):
    """
    Collect all the classlabels of a given model m.

    :param m:           Model
    :param nb_targ:     Number of target attributes
    :return:
    """

    if hasattr(m, 'classes_'):
        if nb_targ == 1:
            # Single-target model
            if type(m.classes_) is list:  # Hotfix. TODO(elia): DO THIS MORE NICELY
                # This occurs when we ask classes_ of a model we built ourselves.
                m_classlabels = m.classes_
            else:
                m_classlabels = [m.classes_]

        else:
            # Multi-target model
            m_classlabels = [m.classes_[j] for j in range(nb_targ)]

    else:
        # If no classlabels are present, we assume a fully numerical model
        #warnings.warn("Model (ID: {}) has no classes_ attribute. Assuming only numerical targets".format(m_idx))
        m_classlabels  = [['numeric']] * nb_targ
        #TODO(elia): This needs to be done nicer.

    return m_classlabels


def fill_in_or_check_clf_labels(clf_labels, m_classlabels, m_targ):
    """
    Update (in case of default value) or check the consistency of the
    given classlabels of the model and its target attributes.

    :param clf_labels:
    :param m_classlabels:
    :param m_targ:
    :return:
    """

    for t_idx, t in enumerate(m_targ):
        if np.array_equal(clf_labels[t], [0]):
            # If the default value of [0] is still there
            clf_labels[t] = m_classlabels[t_idx]
        else:
            # Do a check whether what is provided is consistent
            assert np.array_equal(clf_labels[t], m_classlabels[t_idx])
    return clf_labels


def update_clf_labels(clf_labels, m_classlabels, m_targ):
    """
    Update the classlabels.

    Given an array of clf_labels, for each attribute known to the system,
    add the information on classlabels provided by a single model to it.
    This information is contained in m_classlabels, which are the classlabels
    known by the current model, on the targets as specified in m_targ.

    Update (in case of default value) or expand the present clf_labels.

    Clf_labels thus is own to the MERCS system, and not to the individual
    classifiers.

    :param clf_labels:
    :param m_classlabels:
    :param m_targ:
    :return:
    TODO(elia): Very general problem. We need to deal with numeric attributes in a nicer way.
    """

    for t_idx, t in enumerate(m_targ):
        if np.array_equal(clf_labels[t], [0]):
            # If the default value of [0] is still there, update current
            clf_labels[t] = m_classlabels[t_idx]

        elif np.array_equal(clf_labels[t], ['numeric']):
            assert np.array_equal(m_classlabels[t_idx], ['numeric'])
        else:
            # Join current m_classlabels with those already present
            classlabels_list = [clf_labels[t], m_classlabels[t_idx]]
            clf_labels[t] = join_classlabels(classlabels_list)

    return clf_labels


def join_classlabels(classlabels_list):
    """
    Get the union of the provided classlabels

    This is crucial whenever models are trained on different subsets of the
    data_csv, and have other ideas about what the classlabels are.
    """

    all_classes = np.concatenate(classlabels_list)

    result = np.array(list(set(all_classes)))
    result.sort()

    return result


## Feature Importance
def collect_FI(m_list, m_codes):
    """
    Collect the feature importance of the models.

    :param m_codes:
    :param m_list:
    :return:
    """

    FI = np.zeros(m_codes.shape)
    m_desc, _, _ = codes_to_query(m_codes)

    for mod_i in range(len(m_list)):
        for desc_i, attr_i in enumerate(m_desc[mod_i]):
            FI[mod_i, attr_i] = m_list[mod_i].feature_importances_[desc_i]

    return FI
