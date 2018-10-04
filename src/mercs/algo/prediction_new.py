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

    return mas, aas


def _mi_pred_qry(mas, aas, q_desc, q_targ, m_codes):

    steps = [1]

    # Zero-step
    aas[q_desc] = 0

    for n in steps:
        # Collect available atts/mods
        avl_atts = _available_atts(aas, n)
        avl_mods = _available_mods(mas)

        avl_m_codes = m_codes[avl_mods]

        # Activate atts/mods
        act_atts = _active_atts(q_targ)
        act_mods = _active_mods_mi(avl_atts, act_atts, avl_mods, avl_m_codes)

        aas[act_atts] = n
        mas[act_mods] = n

    return mas, aas


# MA-pred
def ma_pred_algo(m_codes, q_codes, settings):
    assert isinstance(m_codes, np.ndarray)
    assert isinstance(q_codes, np.ndarray)
    assert len(m_codes.shape) == len(q_codes.shape) == 2
    assert m_codes.shape[1] == q_codes.shape[1]

    initial_threshold = settings['param']
    step_size = settings['its']
    assert isinstance(initial_threshold, (int, float))
    assert isinstance(step_size, float)
    assert 0.0 < initial_threshold <= 1.0
    assert 0.0 < step_size < 1.0

    # Preliminaries
    nb_mods, nb_atts, nb_qrys = _extract_global_numbers(m_codes, q_codes)
    q_desc, q_targ, _ = codes_to_query(q_codes)

    thresholds = np.arange(initial_threshold, -1, -step_size)

    mas, aas = _init_mas_aas(nb_mods, nb_atts, nb_qrys)

    for q_idx in range(nb_qrys):
        mas[q_idx], aas[q_idx] = _ma_pred_qry(mas[q_idx],
                                              aas[q_idx],
                                              q_desc[q_idx],
                                              q_targ[q_idx],
                                              m_codes,
                                              thresholds)
        pass

    return mas, aas


def _ma_pred_qry(mas, aas, q_desc, q_targ, m_codes, thresholds):

    steps = [1]

    # Zero-step
    aas[q_desc] = 0

    mas_mi, _ = _mi_pred_qry(mas, aas, q_desc, q_targ, m_codes)
    mas[mas_mi == -1] = 0
    mas[mas_mi == 1] = -1

    for n in steps:
        # Collect available atts/mods
        avl_atts = _available_atts(aas, n)
        avl_mods = _available_mods(mas)

        avl_m_codes = m_codes[avl_mods]

        # Activate atts/mods
        act_atts = _active_atts(q_targ)
        aas[act_atts] = n

        for att in act_atts:
            single_act_att = [att]
            act_mods = _active_mods_ma(avl_atts,
                                       single_act_att,
                                       avl_mods,
                                       avl_m_codes,
                                       thresholds)
            mas[act_mods] = n

    return mas, aas


# MAFI-pred
def mafi_pred_algo(m_codes, q_codes, settings):
    assert isinstance(m_codes, np.ndarray)
    assert isinstance(q_codes, np.ndarray)
    assert len(m_codes.shape) == len(q_codes.shape) == 2
    assert m_codes.shape[1] == q_codes.shape[1]

    initial_threshold = settings['param']
    step_size = settings['its']
    assert isinstance(initial_threshold, (int, float))
    assert isinstance(step_size, float)
    assert 0.0 < initial_threshold <= 1.0
    assert 0.0 < step_size < 1.0

    # TODO: This musn't come packed in 'settings'
    feature_importances = settings['FI']

    # Preliminaries
    nb_mods, nb_atts, nb_qrys = _extract_global_numbers(m_codes, q_codes)
    q_desc, q_targ, _ = codes_to_query(q_codes)

    thresholds = np.arange(initial_threshold, -1, -step_size)

    mas, aas = _init_mas_aas(nb_mods, nb_atts, nb_qrys)

    for q_idx in range(nb_qrys):
        mas[q_idx], aas[q_idx] = _mafi_pred_qry(mas[q_idx],
                                                aas[q_idx],
                                                q_desc[q_idx],
                                                q_targ[q_idx],
                                                m_codes,
                                                thresholds,
                                                feature_importances)

    return mas, aas


def _mafi_pred_qry(mas, aas, q_desc, q_targ, m_codes, thresholds, feature_importances):

    steps = [1]

    # Zero-step
    aas[q_desc] = 0

    mas_mi, _ = _mi_pred_qry(mas, aas, q_desc, q_targ, m_codes)
    mas[mas_mi == -1] = 0
    mas[mas_mi == 1] = -1

    for n in steps:
        # Collect available atts/mods
        avl_atts = _available_atts(aas, n)
        avl_mods = _available_mods(mas)

        avl_m_codes = m_codes[avl_mods]
        avl_f_imprt = feature_importances[avl_mods]

        # Activate atts/mods
        act_atts = _active_atts(q_targ)
        aas[act_atts] = n

        for att in act_atts:
            single_act_att = [att]
            act_mods = _active_mods_mafi(avl_atts,
                                         single_act_att,
                                         avl_mods,
                                         avl_m_codes,
                                         thresholds,
                                         avl_f_imprt)
            mas[act_mods] = n

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


def _active_mods_mi(avl_atts, act_atts, avl_mods, avl_m_codes):
    assert avl_m_codes.shape[0] == avl_mods.shape[0]
    targ_encoding = encode_attribute(1, [0], [1])

    # Calculate appropriateness scores for all available models
    avl_mods_appr_scores = np.zeros(avl_mods.shape[0])
    for m_idx, m_code in avl_m_codes:
        avl_mods_appr_scores[m_idx] = np.sum(m_code[act_atts] == targ_encoding)

    # Activate models with sufficiently high appropriateness scores
    act_mods_idx = np.where(avl_mods_appr_scores >= 1.0)[0]

    assert _assert_all_act_atts_as_targ(act_mods_idx, avl_m_codes, act_atts)
    act_mods = avl_mods[act_mods_idx]

    return act_mods


def _active_mods_ma(avl_atts, act_atts, avl_mods, avl_m_codes, thresholds):
    assert avl_m_codes.shape[0] == avl_mods.shape[0]
    desc_encoding = encode_attribute(0, [0], [1])

    # Calculate appropriateness scores for all available models
    avl_mods_appr_scores = np.zeros(avl_mods.shape[0])
    for m_idx, m_code in avl_m_codes:
        overlap_avl_desc_atts = np.sum(m_code[avl_atts] == desc_encoding)
        total_count_desc_atts = np.sum(m_code[:] == desc_encoding)

        avl_mods_appr_scores[m_idx] = overlap_avl_desc_atts/total_count_desc_atts

    # Activate models with sufficiently high appropriateness scores
    for threshold in thresholds:
        act_mods_idx = np.where(avl_mods_appr_scores >= threshold)[0]

        if _assert_all_act_atts_as_targ(act_mods_idx, avl_m_codes, act_atts):
            break

    act_mods = avl_mods[act_mods_idx]

    return act_mods


def _active_mods_mafi(avl_atts,
                      act_atts,
                      avl_mods,
                      avl_m_codes,
                      thresholds,
                      avl_f_imprt):
    assert avl_m_codes.shape[0] == avl_f_imprt.shape[0] == avl_mods.shape[0]

    # Calculate appropriateness scores for all available models
    avl_mods_appr_scores = np.zeros(avl_mods.shape[0])
    for m_idx, m_code in avl_m_codes:
        avl_mods_appr_scores[m_idx] = np.sum(avl_f_imprt[avl_atts])

    # Activate models with sufficiently high appropriateness scores
    for threshold in thresholds:
        act_mods_idx = np.where(avl_mods_appr_scores >= threshold)[0]

        if _assert_all_act_atts_as_targ(act_mods_idx, avl_m_codes, act_atts):
            break

    act_mods = avl_mods[act_mods_idx]

    return act_mods


# Assert solution
def _assert_all_act_atts_as_targ(act_mods_idx, avl_m_codes, act_atts):
    assert act_mods_idx.shape[0] > 0
    target_encoding = encode_attribute(1, [0], [1])

    filtered_m_codes = avl_m_codes[act_mods_idx, act_atts]
    act_atts_as_targ = np.where(filtered_m_codes == target_encoding)[1]
    check = np.unique(act_atts_as_targ)

    return check.shape[0] == act_atts.shape[0]


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
