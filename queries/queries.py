import json
import os
import numpy as np
from scipy import special

# 1. Query Compilation
def compile_queries(q_keyword='basic', q_param=1, **kwargs):
    """
    Compile attribute sets for induction and inference (or 'queries')

    By query we mean a certain inference task that is required from the model. Such an inference task
    is defined by an input set of attributes and an output set of attributes, both subsets of the total set of attributes of the dataset.

    :param q_keyword        Defines the query generation method
    :param q_param          Some methods of query compilation can take an additional parameter.
    :param atts             This is an array that contains all the attributes of the dataset, e.g. [0, 1, 2, 3]
    :param X                Pandas DataFrame on which the model is trained.
    """

    # Prelims
    if {}.keys() < kwargs.keys() < {'atts', 'nb_atts', 'X'}:
        if kwargs.keys() == {'X'}:
            atts = list(range(len(kwargs['X'].columns)))  # Create atts from the supplied DataFrame
        elif kwargs.keys() == {'nb_atts'}:
            atts = list(range(kwargs['nb_atts']))
        else:
            atts = kwargs['atts']
    else:
        raise ValueError("Exactly one keyword is required:\n"
                         "    1. 'atts'    Array containing idx of attributes\n"
                         "    2. 'nb_atts  Number of attributes\n"
                         "    3. 'X'       Pandas DataFrame of actual data_csv")

    induction_desc, induction_targ = atts, atts

    # Query Compilation
    if q_keyword == 'basic':
        q_desc, q_targ, q_miss = basic_query_algo(atts)
    elif q_keyword == 'double':
        q_desc, q_targ, q_miss = double_query_algo(atts)
    elif q_keyword == 'it_targetsets':
        q_desc, q_targ, q_miss = it_targetsets_query_algo(atts)
    elif q_keyword == 'it_missing':
        q_desc, q_targ, q_miss = it_missing_query_algo(atts)
    elif q_keyword == 'missing_ratio':
        q_desc, q_targ, q_miss = missing_ratio_query_algo(atts, q_param)
    elif q_keyword == 'it_missing_2':
        q_desc, q_targ, q_miss = it_missing_2_query_algo(atts, q_param)
    else:
        raise ValueError("Did not recognize keyword: {}\n"
                         "Accepted keywords are (basic, double, it_targetsets, "
                         "it_missing, missing_ratio, it_missing_2)".format(q_keyword))

    return induction_desc, induction_targ, q_desc, q_targ, q_miss


def basic_query_algo(atts):
    """
    Leave-one-out prediction, with a maximum of 50 queries.

    :param atts:
    :return:
    """

    query_targ_sets = [[i] for i in atts if i < 50]

    # All the remaing atts are used as inputs
    query_desc_sets = [list(set(atts) - set(qts)) for qts in query_targ_sets]
    query_miss_sets = [[] for i in query_desc_sets]
    return query_desc_sets, query_targ_sets, query_miss_sets


def double_query_algo(atts):
    """
    Leave two out prediction.

    :param atts:
    :return:
    """

    query_targ_sets = [[i - 1, i] for i in atts if i > 0 if i < 100]

    # All the remaining atts are used as inputs
    query_desc_sets = [list(set(atts) - set(qts)) for qts in
                       query_targ_sets]
    query_miss_sets = [[] for i in query_desc_sets]

    return query_desc_sets, query_targ_sets, query_miss_sets


def it_targetsets_query_algo(atts):
    """
    We build targetsets of different sizes.

    We specify the maximum amount of different sizes, and we specify an
    upper limit. This limit is a percentage of the total.

    Per size we specify the maximum amount of queries.
    This can be a difficult function because of the comb function,
    this might be left out.
    """

    nb_atts = len(atts)

    # In total we want to ask no more than: max_nb_ts_sizes*max_nb_queries
    max_nb_ts_sizes = 20    # Amount of different targetset sizes to check
    max_nb_queries = 20     # Amount of queries per targetset size
    ratio_t_to_tot = 0.4    # Percentage of nb_atts to predict in one query
    max_ts_size = nb_atts * ratio_t_to_tot  # Largest targetset allowed

    # linspace + unique takes care of never taking too much different sizes
    targetset_sizes = np.unique(np.linspace(1, max_ts_size, num=max_nb_ts_sizes,
                                            dtype=int)).tolist()

    query_targ_sets, query_desc_sets = [], []
    for ts_size in targetset_sizes:
        temp_query_targ_sets = []
        combs = int(special.comb(nb_atts, ts_size))

        # If the maximum number of queries cannot be computed, we settle for all the possible ones.
        nb_queries = np.min([max_nb_queries,
                             combs])

        while len(temp_query_targ_sets) < nb_queries:
            targetset = set(np.random.choice(atts,
                                             replace=False,
                                             size=ts_size).tolist())
            if targetset not in temp_query_targ_sets:
                temp_query_targ_sets.append(targetset)

        temp_query_targ_sets = [list(i) for i in temp_query_targ_sets]

        # No missing values
        temp_query_desc_sets = [list(set(atts) - set(qts))
                                for qts in temp_query_targ_sets]

        query_targ_sets.extend(temp_query_targ_sets)
        query_desc_sets.extend(temp_query_desc_sets)

    query_miss_sets = [[] for i in query_desc_sets]
    return query_desc_sets, query_targ_sets, query_miss_sets


def it_missing_query_algo(atts):
    """
    Build missingsets of different sizes.

    Per size we ask a fixed amount of queries
    """

    nb_atts = len(atts)

    # In total we want to ask no more than: max_nb_ms_size*max_nb_queries
    max_nb_queries = 5  # Amount of missingsets we generate PER size. Each different missingset results in a different query, hence the name.

    max_nb_ms_sizes = 20  # Amount of different missingset sizes we want to check
    ratio_m_to_tot = 0.4
    max_ms_size = nb_atts * ratio_m_to_tot  # Largest missingset that we allow (in absolute terms)

    missingset_sizes = np.unique(np.linspace(1, max_ms_size, num=max_nb_ms_sizes,
                                             dtype=int)).tolist()  # linspace + unique takes care of never taking too much different sizes

    query_targ_set = np.random.choice(atts, replace=False, size=1).tolist()

    query_miss_sets, query_desc_sets = [], []
    for ms_size in missingset_sizes:
        available_atts = list(set(atts) - set(query_targ_set))
        temp_query_miss_sets = []
        combs = int(special.comb(nb_atts, ms_size))
        nb_queries = np.min([max_nb_queries,
                             combs])  # If the maximum number of queries cannot be computed, we settle for all the possible ones.

        while len(temp_query_miss_sets) < nb_queries:
            missingset = set(np.random.choice(available_atts, replace=False, size=ms_size).tolist())
            if missingset not in temp_query_miss_sets: temp_query_miss_sets.append(missingset)

        temp_query_miss_sets = [list(i) for i in temp_query_miss_sets]
        temp_query_desc_sets = [list(set(available_atts) - set(qms)) for qms in
                                temp_query_miss_sets]  # No missing values

        query_miss_sets.extend(temp_query_miss_sets)
        query_desc_sets.extend(temp_query_desc_sets)

    query_targ_sets = [query_targ_set for i in query_desc_sets]
    return query_desc_sets, query_targ_sets, query_miss_sets


def missing_ratio_query_algo(atts, q_param):
    """
    We build missingsets of a fixed size, given by q_param
    """

    nb_atts = len(atts)

    max_nb_queries = 50  # Amount of missingsets we generate
    ratio_m_to_tot = q_param
    ms_size = int(round(nb_atts * ratio_m_to_tot))  # Size of the missingset, based on the ratio provided as param

    query_targ_set = np.random.choice(atts, replace=False, size=1).tolist()

    query_miss_sets, query_desc_sets = [], []

    available_atts = list(set(atts) - set(query_targ_set))
    combs = int(special.comb(nb_atts, ms_size))
    nb_queries = np.min([max_nb_queries,
                         combs])  # If the maximum number of queries cannot be computed, we settle for all the possible ones.

    while len(query_miss_sets) < nb_queries:
        missingset = set(np.random.choice(available_atts, replace=False, size=ms_size).tolist())
        if missingset not in query_miss_sets: query_miss_sets.append(missingset)

    query_miss_sets = [list(i) for i in query_miss_sets]
    query_desc_sets = [list(set(available_atts) - set(qms)) for qms in query_miss_sets]  # No missing values

    query_targ_sets = [query_targ_set for i in query_desc_sets]
    return query_desc_sets, query_targ_sets, query_miss_sets


def it_missing_2_query_algo(atts, q_param):
    """
    We randomly select some attributes which will be targets.

    Each target is predicted in one query, all the rest is descriptive. Then we gradually leave out descriptive attributes.
    """

    nb_atts = len(atts)

    query_desc_sets, query_targ_sets, query_miss_sets = [], [], []
    nb_diff_configs = 50  # This is an arbitrary parameter!

    np.random.seed(997)
    shuffled_atts = list(np.random.permutation(atts).tolist() for i in range(nb_diff_configs))

    orig_code = generate_query_code(nb_atts, miss_size=0)
    ms_steps = generate_ms_steps(orig_code, q_param)  # Query param determines the amount of steps.
    code = orig_code.copy()

    for atts_shuffle in shuffled_atts:

        # Getting the query, and saving it
        desc, targ, miss = code_to_query(code, atts_shuffle)  # Generate the actual query (3 arrays)
        query_desc_sets.append(desc), query_targ_sets.append(targ), query_miss_sets.append(miss)

        for step in ms_steps:
            code = desc_to_miss(code, step)  # Update code

            # Getting the query, and saving it
            desc, targ, miss = code_to_query(code, atts_shuffle)
            query_desc_sets.append(desc), query_targ_sets.append(targ), query_miss_sets.append(miss)

        code = orig_code.copy()  # Do not forget to reset the code to its original form.
    return query_desc_sets, query_targ_sets, query_miss_sets


# 2. I/O Functionalities
def load_queries(filename=None):
    """
    Load queries from a given file.

    This file is a .json file. It should contain a dict.
    """

    # Load file
    if filename is None: filename = 'queries.json'
    with open(filename) as f: q_dict = json.load(f)

    # Extract useful information
    q_desc = q_dict['queries']['desc_sets']
    q_targ = q_dict['queries']['targ_sets']
    q_miss = q_dict['queries']['miss_sets']

    assert len(q_targ) == len(q_desc) == len(q_miss)  # Sanity check

    return q_desc, q_targ, q_miss


def make_queries_dict(q_desc,
                      q_targ,
                      q_miss,
                      q_keyword='Unknown',
                      q_param='Unknown'):
    """
    Create a dict containing the relevant information about the queries

    :param q_desc:
    :param q_targ:
    :param q_miss:
    :param q_keyword:
    :param q_param:
    :return:
    """

    d = {}
    d['queries'] = {'desc_sets':    q_desc,
                    'targ_sets':    q_targ,
                    'miss_sets':    q_miss,
                    'keyword':      q_keyword,
                    'param':        q_param}
    return d


def save_queries(q_dict, filename=None):
    """
    Save generated queries to a file with a given filename.

    :param q_dict:    Dict of queries
    :param filename:        Optional filename (Default: queries.json in cwd)
    :return:
    """
    if filename is None: filename = 'queries.json'

    with open(filename, 'w') as f:
        json.dump(q_dict, f)
    return


def check_given_query_codes(nb_atts, **kwargs):
    """
    Generate query codes for prediction.

    Based on optional keyword arguments, generate query codes for prediction.
    Needs the amount of attributes to run checks.

    What this does is that it ensures the correct format to do prediction,
    and runs basic check.

    :param kwargs:      1. query_code/q_code
                        2. query_codes/q_codes

                        If this is provided, these query_codes are returned.
    :return:
    """

    relevant_keywords = kwargs.keys() & {'q_code', 'query_code', 'q_codes', 'query_codes'}

    for k in relevant_keywords:
        if k in {'q_code', 'query_code'}:
            assert len(kwargs[k]) == nb_atts
            q_codes = [kwargs[k]]               # Single code put in array

            return q_codes
        elif k in {'q_codes', 'query_codes'}:
            assert len(kwargs[k][0]) == nb_atts
            q_codes = kwargs[k]                 # Multiple codes left untouched

            return q_codes
        else:
            raise ValueError("Could not generate query codes."
                             "Please provide one of the following:"
                             "- query_code"
                             "- query_codes"
                             "- q_code"
                             "- q_codes")


# 3. Helper functions
def generate_default_query_code(nb_atts):
    """
    Generate default query codes array.

    This means a q_codes, containing a single q_code array, which means:
        1.  len(q_codes) = 1
        2.  len(q_codes[0]) = nb_atts

    The default query thus assumes all attributes known,
    except the last one, which serves as target.

    :param nb_atts:     Number of attributes in the dataset.
    :return:
    """
    assert isinstance(nb_atts, int)

    q = [0] * nb_atts
    q[-1] = 1
    q_codes = [q]
    return q_codes


def generate_query_code(nb_atts,
                        desc_size=None,
                        targ_size=1,
                        miss_size=None):
    """
    Generate an array of the same length as the original attribute array.

    The query code means the following:
        0:  Descriptive attribute
        1:  Target attribute
        -1: Missing attribute

    TODO: Switch to more general code + Centralize all coding affairs.
    """

    # Some automatic procedures if we are given some freedom.
    if desc_size == None and miss_size == None:
        desc_size = nb_atts - targ_size  # All the rest is assumed descriptive
        miss_size = 0
    elif desc_size != None and miss_size == None:
        miss_size = nb_atts - targ_size - desc_size
    elif desc_size == None and miss_size != None:
        desc_size = nb_atts - targ_size - miss_size
    else:
        pass

    # Bad user choices can still cause meaningless queries, so we check
    assert ((desc_size + targ_size + miss_size) == nb_atts)
    assert (targ_size > 0)
    assert (desc_size > 0)

    # The actual encoding
    code = [0 for i in range(desc_size)]
    code.extend([+1 for i in range(targ_size)])
    code.extend([-1 for i in range(miss_size)])
    return code


def desc_to_miss(code, amount=1):
    """
    Change the first 'amount' of 0's to -1's
    """

    changes, i = 0, 0

    while (i < len(code) and changes < amount):
        if code[i] == 0:
            code[i] = -1
            changes += 1
        else:
            pass
        i += 1
    return code


def generate_ms_steps(code, param):
    """
    Generate an array that contains the (integer) steps in which we increase the set of missing attributes.

    This is in the context of converting descriptive attributes to missing ones.

    :param param -  If parameter is a float < 1, it is interpreted as the percentage increase that is desired
                    If parameter is a int > 1, it is interpreted as the amount of attributes that needs to be added
                    to the missing set at each step.
    :param code - The starting code
    """

    max_amount_steps = int(1 / param) if param < 1 else int(param)
    available_atts = code.count(0) - 1  # We have to keep at least one descriptive attribute, hence -1!

    ms_sizes = np.linspace(0, available_atts, num=max_amount_steps + 1, dtype=int).tolist()
    ms_steps = [ms_sizes[i] - ms_sizes[i - 1] for i in range(1, len(ms_sizes)) if (ms_sizes[i] - ms_sizes[i - 1] > 0)]

    return ms_steps


def code_to_query(code, atts=None):
    """
    Change the code-array to an actual query, which are three arrays.

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

    desc = [x for i, x in enumerate(atts) if code[i] == 0]
    targ = [x for i, x in enumerate(atts) if code[i] == 1]
    miss = [x for i, x in enumerate(atts) if code[i] == -1]
    return desc, targ, miss


# 4. Access to demos
def load_example_queries(dataset):
    """
    Load some example queries.

    :param dataset:    Dataset name.
    :return:                Desc sets, Target sets, Missing sets. (i.e.: Sets of attributes.)
    """

    current_dir = os.path.dirname(__file__)
    queries_dir = os.path.join(current_dir, 'query_json')

    filename = os.path.join(queries_dir, dataset + '.json')

    query_desc_sets, query_targ_sets, query_miss_sets = load_queries(filename=filename)
    return query_desc_sets, query_targ_sets, query_miss_sets


def load_nursery_queries():
    """
    Load some example queries that are appropriate for the nursery dataset,

    :return:
    """

    query_desc_sets, query_targ_sets, query_miss_sets = load_example_queries('nursery')
    return query_desc_sets, query_targ_sets, query_miss_sets


def load_netflix_queries():
    """
    Load some example queries that are appropriate for the netflix dataset,

    :return:
    """

    query_desc_sets, query_targ_sets, query_miss_sets = load_example_queries('netflix')
    return query_desc_sets, query_targ_sets, query_miss_sets


def load_jester_queries():
    """
    Load some example queries that are appropriate for the jester dataset,

    :return:
    """

    query_desc_sets, query_targ_sets, query_miss_sets = load_example_queries('jester')
    return query_desc_sets, query_targ_sets, query_miss_sets
