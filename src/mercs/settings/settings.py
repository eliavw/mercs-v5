from ..utils.encoding import codes_to_query
from ..utils.classlabels import collect_and_verify_clf_classlabels
from ..utils.metadata import collect_FI


# Main methods
def create_settings():
    """
    Quickly generate a default settings dictionary for a MERCS experiment.

    :return:    Dict of settings

    TODO: Here the key is 'queries' in exp it is 'queries'. Make this always right!
    """

    settings={}

    settings['induction'] = {'type':    'DT'}

    settings['selection'] = {'type':    'Base',
                             'its':     1,
                             'param':   1}

    settings['prediction'] = {'type':   'MI',
                              'its':    0.1,
                              'param':  1}

    settings['queries'] = {}

    settings['metadata'] = {}

    settings['model_data'] = {}

    return settings


def filter_kwargs_update_settings(s, prefix=None, delimiter='_', **kwargs):

    param_map = compile_param_map(prefix=prefix, delimiter=delimiter, **kwargs)
    return update_settings_dict(s, param_map, **kwargs)


def update_meta_data(s, m_list, m_codes):
    """
    Update the metadata of this MERCS object.

    This concerns info about the models themselves.
    N.b.:   This can only be done AFTER selection and training took place,
            because things like the classlabels are required.

    Parameters
    ----------
    s: dict
        Settings dictionary
    m_list: list
        List containing all the component models
    m_codes
        List containing the codes of the component models

    Returns
    -------

    """

    s['clf_labels'] = collect_and_verify_clf_classlabels(m_list, m_codes)
    s['FI'] = collect_FI(m_list, m_codes)

    return s


def update_query_settings(s, nb_atts, delimiter='_', **kwargs):

    param_map = compile_param_map(prefix='qry', delimiter=delimiter, **kwargs)
    relevant_kwargs = {v:kwargs[k] for k, v in param_map.items()}

    if 'codes' in relevant_kwargs:
        # Check codes and if they do not comply replace by default
        codes = relevant_kwargs['codes']
        if len(codes[0]) == nb_atts:
            s['codes'] = codes
        else:
            s['codes'] = generate_default_query_code(nb_atts)

        s['q_desc'], s['q_targ'], s['q_miss'] = codes_to_query(s['codes'])

    elif 'code' in relevant_kwargs:
        # Wrap single code is extra array
        codes = [relevant_kwargs['code']]
        print("SETTINGS.PY: I AM READING A SINGLE QUERY CODE, I.E: {}".format(relevant_kwargs['code']))
        update_query_settings(s, nb_atts, qry_codes=codes) # Do NOT pass the delimiter here!
    else:
        # Check what is already present
        codes = s.get('codes', None)

        if codes is None:
            s['codes'] = generate_default_query_code(nb_atts)
        elif len(codes[0]) != nb_atts:
            s['codes'] = generate_default_query_code(nb_atts)
        else:
            assert len(codes[0]) == nb_atts
            s['codes'] = codes

        s['q_desc'], s['q_targ'], s['q_miss'] = codes_to_query(s['codes'])

    return s


# Helpers
def update_settings_dict(settings, param_map, **kwargs):
    """
    Update a dictionary through a parameter map.

    :param settings:        Dict that needs updating
    :param param_map:       Dict that maps keyword arguments to
                            their entries in the dict
    :return:                A newer version of the original dict.
    """

    for k in kwargs:
        if k in param_map:
            settings[param_map[k]] = kwargs[k]
        else:
            pass

    return settings


def compile_param_map(prefix=None, delimiter='_', **kwargs):
    """
    Compile parameter map automatically.

    This is often very useful.

    :param prefix:
    :param kwargs:
    :return:
    """
    if prefix is not None:
        prefix += delimiter
    else:
        prefix=''

    param_map = {k: k.split(prefix)[1]
                 for k in kwargs
                 if k.startswith(prefix)}

    return param_map


def generate_default_query_code(nb_atts):
    """
    Generate default queries codes array.

    This means a q_codes, containing a single q_code array, which means:
        1.  len(q_codes) = 1
        2.  len(q_codes[0]) = nb_atts

    The default queries thus assumes all attributes known,
    except the last one, which serves as target.

    :param nb_atts:     Number of attributes in the dataset.
    :return:
    """
    assert isinstance(nb_atts, int)

    q = [0] * nb_atts
    q[-1] = 1
    q_codes = [q]
    return q_codes