import numpy as np

from ..utils.encoding import codes_to_query, encode_attribute

from ..utils.debug import debug_print
VERBOSITY = 0


# MI-pred
def mi_pred_algo(m_codes, q_codes):
    assert isinstance(m_codes, np.ndarray)
    assert isinstance(q_codes, np.ndarray)
    assert len(m_codes.shape) == len(q_codes.shape) == 2
    assert m_codes.shape[1] == q_codes.shape[1]

    # Preliminaries
    nb_mods, nb_atts, nb_qrys = _extract_global_numbers(m_codes, q_codes)
    q_desc, q_targ, _ = codes_to_query(q_codes)

    mas, aas = _init_mas_aas(nb_mods, nb_atts, nb_qrys)

    for q_idx in range(nb_qrys):
        mas[q_idx], aas[q_idx] = _mi_pred_qry(mas[q_idx],
                                              aas[q_idx],
                                              q_desc[q_idx],
                                              q_targ[q_idx],
                                              m_codes)
        pass

    return mas, aas


def _mi_pred_qry(mas, aas, q_desc, q_targ, m_codes):

    steps = list(range(1,2))

    # Zero-step
    aas[q_desc] = 0

    for n in steps:
        # Collect available atts/mods
        avl_atts = _available_atts(aas, n)
        avl_mods = _available_mods(mas)

        avl_m_codes = m_codes[avl_mods]

        # Activate atts/mods
        act_atts = _active_atts(q_targ)
        act_mods = _active_mods(avl_atts, act_atts, avl_mods, avl_m_codes)

        aas[act_atts] = 1
        mas[act_mods] = 1

    return mas, aas


# Four steps
def _available_atts(aas, step):
    """
    Available attributes were 'active' in a previous step.

    I.e.:

    aas == 0:
        Attribute was always available, i.e., descriptive attribute
        of the query.
    1 <= aas < step:
        Attribute was activated in a previous step of the prediction
        algorithm, and thus available

    Parameters
    ----------
    aas
    step

    Returns
    -------

    """
    return np.where((0 <= aas) & (aas < step))[0]


def _available_mods(mas):
    """
    Available models are models that have not been 'active' before

    Parameters
    ----------
    mas
    step

    Returns
    -------

    """
    return np.where(mas == -1)[0]


def _active_atts(q_targ):
    return np.array(q_targ)


def _active_mods(avl_atts, act_atts, avl_mods, avl_m_codes):
    assert avl_m_codes.shape[0] == avl_mods.shape[0]
    targ_encoding = encode_attribute(1, [0], [1])

    act_mods_idx = np.where(avl_m_codes[:, act_atts] == targ_encoding)[0]
    act_mods = avl_mods[act_mods_idx]

    return act_mods


# Initialize
def _extract_global_numbers(m_codes, q_codes):
    nb_mods, nb_atts = m_codes.shape
    nb_qrys = q_codes.shape[0]
    return nb_mods, nb_atts, nb_qrys


def _init_mas_aas(nb_models, nb_atts, nb_queries):
    """
    Initialize the mas and aas arrays.

    Difference with the above is that this outputs full numpy arrays.

    Parameters
    ----------
    nb_models: int
        Total number of models in the MERCS model
    nb_atts: int
        Total number of attributes in the MERCS model
    nb_queries: int
        Total number of queries that needs to be answered.

    Returns
    -------
    mas: np.ndarray, shape (nb_queries, nb_models)
        E.g.:
            np.array([[0,0,0],
                      [0,0,0]])
    aas: np.ndarray, shape (nb_queries, nb_atts)
        E.g.:
            np.array([[-1,-1,-1,-1],
                      [-1,-1,-1,-1]])
    """

    assert isinstance(nb_models, int)
    assert isinstance(nb_atts, int)
    assert isinstance(nb_queries, int)

    mas = np.full((nb_queries, nb_models), -1, dtype=np.int)
    aas = np.full((nb_queries, nb_atts), -1, dtype=np.int)
    return mas, aas
