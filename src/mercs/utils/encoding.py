
def codes_to_query(codes, atts=None):
    """
    Change the codes-array to an actual queries, which are three arrays.

    :param code:                Array that contains: 0-desc/1-target/-1-missing code for each attribute
    :param atts:                Array that contains the attributes (indices)
    :return: Three arrays.      One for desc atts indices, one for targets, one for missing
    """

    if atts is None:
        atts = list(range(len(codes[0])))
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
    """

    if atts is None:
        atts = list(range(len(code)))
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