from ..utils.encoding import codes_to_query, encode_attribute
from ..utils.classlabels import collect_and_verify_clf_classlabels
from ..utils.metadata import collect_feature_importances

from ..utils.debug import debug_print
VERBOSITY = 0


# Main methods
def create_settings():
    """
    Generate a default settings dictionary for a MERCS model.

    Returns
    -------
    settings: dict
        Dictionary of default settings
    """

    settings = {}

    settings['induction'] = {'type':    'DT'}

    settings['selection'] = {'type':    'Base',
                             'its':     1,
                             'param':   1}

    settings['prediction'] = {'type':   'MI',
                              'its':    0.1,
                              'param':  0.95}

    settings['queries'] = {}

    settings['metadata'] = {}

    settings['model_data'] = {}

    return settings


def filter_kwargs_update_settings(s, prefix=None, delimiter='_', **kwargs):

    param_map = _compile_param_map(prefix=prefix, delimiter=delimiter, **kwargs)
    return _update_settings_dict(s, param_map, **kwargs)


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
    s['FI'] = collect_feature_importances(m_list, m_codes)

    return s


def update_query_settings(s, nb_atts, delimiter='_', **kwargs):

    param_map = _compile_param_map(prefix='qry', delimiter=delimiter, **kwargs)
    relevant_kwargs = {v: kwargs[k] for k, v in param_map.items()}

    if 'codes' in relevant_kwargs:
        # Check codes and if they do not comply replace by default
        codes = relevant_kwargs['codes']

        if _verify_decent_query_codes(codes, nb_atts):
            s['codes'] = codes
        else:
            s['codes'] = _generate_default_query_code(nb_atts)

        s['q_desc'], s['q_targ'], s['q_miss'] = codes_to_query(s['codes'])

    elif 'code' in relevant_kwargs:
        # Wrap single code in extra array for consistency
        msg = """
                In file:\t\t\t{}\n
                I am reading a single query code, i.e.:\t{}\n
                """.format(__file__, relevant_kwargs['code'])
        debug_print(msg, V=VERBOSITY)

        codes = [relevant_kwargs['code']]
        update_query_settings(s, nb_atts, qry_codes=codes)  # N.B.: Do NOT pass the delimiter here!
    else:
        # Nothing provided in kwargs, we check what is already present.
        codes = s.get('codes', None)
        update_query_settings(s, nb_atts, qry_codes=codes) # N.B.: Do NOT pass the delimiter here!

    return s


# Helpers
def _update_settings_dict(settings, param_map, **kwargs):
    """
    Update settings dictionary through a parameter map.

    Parameters
    ----------
    settings: dict
        Settings dictionary which needs updating
    param_map: dict
        Mapping from keys in kwargs (keyword arguments) to
        their corresponding keys in the settings dictionary.
    kwargs

    Returns
    -------

    The original settings dictionary with potentially some entries with
    new values.

    """

    for k in kwargs:
        if k in param_map:
            settings[param_map[k]] = kwargs[k]
        else:
            pass

    return settings


def _compile_param_map(prefix=None, delimiter='_', **kwargs):
    """
    Automatically compile parameter map.

    This is best explained by an example;

        in = {'character_name':         'obi-wan',
              'character_status':       'jedi-master',
              'movie_title':            'a new hope'}

        out = _compile_param_map(prefix='character', **in)

        out = {'character_name':    'name',
               'character_status':  'status'}

    Parameters
    ----------
    prefix: str
        Prefix that has to be cut off.
    delimiter: str
        Delimiter that separates the prefix from the rest
    kwargs: dict
        Dictionary of keyword arguments that may contain the prefix.
        These have to be mapped.

    Returns
    -------

    """

    if prefix is not None:
        prefix += delimiter
    else:
        prefix = ''

    param_map = {k: k.split(prefix)[1]
                 for k in kwargs
                 if k.startswith(prefix)}

    return param_map


def _generate_default_query_code(nb_atts):
    """
    Generate default query-code array.

    This generating a q_codes array, which contains a single
    q_code array. Concretely, this means;
        1. len(q_codes) = 1
        2. len(q_codes[0]) = nb_atts

    The default query code (q_code) assumes:
        1. The last attribute is the target attribute
        2. All the other attributes are known and thus descriptive
        3. (Follows from the above) No missing attributes

    Parameters
    ----------
    nb_atts: int
        Number of attributes in the dataset

    Returns
    -------

    """
    assert isinstance(nb_atts, int)
    assert 2 <= nb_atts

    desc_code = encode_attribute(0,[0],[1])
    targ_code = encode_attribute(1,[0],[1])

    q = [desc_code] * nb_atts   # Mark all attributes as descriptive
    q[-1] = targ_code           # Mark last attribute as target
    q_codes = [q]               # Wrap in list for uniformity

    return q_codes


def _verify_decent_query_codes(codes, nb_atts):

    if codes is None:
        result = False
    else:
        result = _check_all_lengths(codes, nb_atts)

    return result


def _check_all_lengths(codes, nb_atts):

    errors = [1 for code in codes
              if len(code) != nb_atts]
    check = len(errors) > 0

    return check
