import numpy as np
import warnings


def codes_to_query(codes, attributes=None):
    """
    Split codes array in three arrays

    Parameters
    ----------
    codes: np.ndarray, shape (nb_codes, nb_attributes)
        Two-dimensional numpy array of codes. Each code encodes the function
        of the attributes.
    attributes: np.ndarray, shape (nb_attributes, ), default=None
        Numpy array that contains the indices of the attributes whose function
        is encoded in the codes. If None, we assume that the attributes indices
        are simply np.arange(nb_attributes)

    Returns
    -------

    """

    assert isinstance(codes, np.ndarray)
    assert isinstance(attributes, (np.ndarray, type(None)))

    nb_codes, nb_atts = codes.shape
    if attributes is None:
        attributes = np.arange(nb_atts)

    desc, targ, miss = [], [], []

    for c_idx in range(nb_codes):
        code = codes[c_idx]
        c_desc, c_targ, c_miss = code_to_query(code, attributes)

        desc.append(c_desc)
        targ.append(c_targ)
        miss.append(c_miss)

    return desc, targ, miss


def code_to_query(code, attributes=None):
    """
    Split the code array into three arrays of attributes of distinct function.

    Parameters
    ----------
    code: np.ndarray, shape (nb_attributes, )
        One-dimensional numpy array that encodes a query. Each entry encodes
        the function of the associated attribute
    attributes: np.ndarray, shape (nb_attributes, ), default=None
        Numpy array that contains the indices of the attributes whose function
        is encoded in the codes. If None, we assume that the attributes indices
        are simply np.arange(nb_attributes)

    Returns
    -------

    """

    assert isinstance(code, np.ndarray)

    nb_atts = code.shape[0]
    if attributes is None:
        attributes = np.arange(nb_atts)
    assert isinstance(attributes, np.ndarray)
    assert code.shape == attributes.shape

    desc_encoding = encode_attribute(0, [0], [1])
    targ_encoding = encode_attribute(1, [0], [1])
    miss_encoding = encode_attribute(2, [0], [1])

    desc, targ, miss = [], [], []

    for i, x in enumerate(attributes):
        if code[i] == desc_encoding:
            desc.append(x)
        elif code[i] == targ_encoding:
            targ.append(x)
        elif code[i] == miss_encoding:
            miss.append(x)
        else:
            msg = """
            Did not recognize encoding: {}\n
            This occured in code: {}\n
            Ignoring this entry.
            """.format(code[i], code)
            warnings.warn(msg)

    return desc, targ, miss


def queries_to_codes(q_desc, q_targ, q_miss, atts=None):
    assert len(q_desc) == len(q_targ) == len(q_miss)
    nb_queries = len(q_desc)

    if atts is None:
        atts = determine_atts(q_desc[0], q_targ[0], q_miss[0])

    codes = [query_to_code(q_desc[i], q_targ[i], q_miss[i], atts=atts)
             for i in range(nb_queries)]

    return codes


def query_to_code(q_desc, q_targ, q_miss, atts=None):
    if atts is None:
        atts = determine_atts(q_desc, q_targ, q_miss)

    code = [encode_attribute(a, q_desc, q_targ) for a in atts]

    return code


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
