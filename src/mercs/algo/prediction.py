from ..utils.utils import *
import numpy as np


# Main Functionalities
def mi_pred_algo(m_codes, q_codes):
    # Preliminaries
    nb_models, nb_atts, nb_queries, m_desc, m_targ, q_desc, q_targ = pred_prelims(m_codes,
                                                                                  q_codes)
    mas, aas = init_mas_aas(nb_models, nb_atts, nb_queries)

    # Building codes
    for q_idx, q_code in enumerate(q_codes):
        # Prelims
        aas[q_idx][q_desc[q_idx]] = 0

        # Model activation
        relevant_models = np.where(m_codes[:, q_targ[q_idx]] == 1)[0] # Models sharing target with queries
        mas[q_idx][relevant_models] = 1

        # Att. activation
        aas[q_idx][q_targ[q_idx]] = 1  # Does not depend on model activation strategy

    return np.array(mas), np.array(aas)


def ma_pred_algo(m_codes, q_codes, settings):
    # Preliminaries
    nb_models, nb_atts, nb_queries, m_desc, m_targ, q_desc, q_targ = pred_prelims(m_codes,
                                                                                  q_codes)
    mas, aas = init_mas_aas(nb_models, nb_atts, nb_queries)

    initial_threshold = settings['param']
    step_size = settings['its']
    assert isinstance(initial_threshold, (int,float))
    assert isinstance(step_size, float)

    thresholds = np.arange(initial_threshold, -1, -step_size)

    # Building codes
    for q_idx, q_code in enumerate(q_codes):
        # Prelims
        aas[q_idx][q_desc[q_idx]] = 0

        relevant_models = np.where(m_codes[:, q_targ[q_idx]] == 1)[0]  # Models that share a target with the queries.
        mas[q_idx][relevant_models] = 1

        avl_mods = mas[q_idx] > 0
        avl_atts = aas[q_idx] > -1

        # Att. activation
        aas[q_idx][q_targ[q_idx]] = 1  # Does not depend on model activation strategy

        # Model activation
        mod_appr_scores = [np.sum(avl_atts.take(m_desc[m_ind])) / len(m_desc[m_ind])
                           if (avl_mods[m_ind] == 1) else -1
                           for m_ind in range(nb_models)]

        for thr in thresholds:
            mas[q_idx] = [1 if (mod_appr_scores[m_ind] > thr) else 0
                                    for m_ind in range(nb_models)]  # All the models that are appropriate enough

            if np.sum(mas[q_idx]) >= 1:
                break  # We demand at least one model

    return np.array(mas), np.array(aas)


def mafi_pred_algo(m_codes, q_codes, settings):
    # Preliminaries
    nb_models, nb_atts, nb_queries, m_desc, m_targ, q_desc, q_targ = pred_prelims(m_codes,
                                                                                  q_codes)
    mas, aas = init_mas_aas(nb_models, nb_atts, nb_queries)

    initial_threshold= settings['param']
    step_size = settings['its']
    assert isinstance(initial_threshold, (int,float))
    assert isinstance(step_size, float)

    thresholds = np.arange(initial_threshold, -1, -step_size)
    FI = settings['FI']

    # Building codes
    for q_idx, q_code in enumerate(q_codes):
        # Prelims
        aas[q_idx][q_desc[q_idx]] = 0

        relevant_models = np.where(m_codes[:, q_targ[q_idx]] == 1)[0]  # Models that share a target with the queries.
        mas[q_idx][relevant_models] = 1

        avl_mods = mas[q_idx] > 0  # Avl. models share a target with queries
        avl_atts = aas[q_idx] > -1

        # Att. activation
        aas[q_idx][q_targ[q_idx]] = 1  # Does not depend on model activation strategy

        # Model activation
        mod_appr_scores = [np.dot(avl_atts, FI[m_ind])
                           if (avl_mods[m_ind] == 1) else -1
                           for m_ind in range(nb_models)]

        for thr in thresholds:
            mas[q_idx] = [1 if (mod_appr_scores[m_ind] > thr) else 0
                                    for m_ind in range(nb_models)]  # All the models that are appropriate enough

            if np.sum(mas[q_idx]) >= 1: break  # We demand at least one model

    return np.array(mas), np.array(aas)


def it_pred_algo(m_codes, q_codes, settings):
    # Preliminaries
    nb_models, nb_atts, nb_queries, m_desc, m_targ, q_desc, q_targ = pred_prelims(m_codes,
                                                                                  q_codes)
    mas, aas = init_mas_aas(nb_models, nb_atts, nb_queries)

    # Collecting parameters from s
    init_threshold = 1
    step_size = settings['param']
    max_layers = settings['its']
    assert isinstance(max_layers, int)
    assert isinstance(step_size, float)

    FI = settings['FI']

    # Building codes
    for q_idx, q_code in enumerate(q_codes):

        # Prelims
        aas[q_idx][q_desc[q_idx]] = 0

        thresholds = iter(np.arange(init_threshold, -1, -step_size))  # Set thresholds
        done = False
        step = 1
        while not done:
            avl_mods = mas[q_idx] > 0
            avl_atts = aas[q_idx] > -1

            # Model activation
            mod_appr_scores = [np.dot(avl_atts, FI[m_idx])
                               if (avl_mods[m_idx] == 0) else -1
                               for m_idx in range(nb_models)]  # Only score non-available ones!

            mod_progress = False
            while not mod_progress:
                thr = next(thresholds)
                # print('Threshold: '+str(thr))
                mas[q_idx] = [step if (mod_appr_scores[m_idx] > thr) else v
                                        for m_idx, v
                                        in enumerate(mas[q_idx])]  # Appropriate enough models
                mas[q_idx] = np.array(mas[q_idx])
                mod_progress = np.sum(mas[q_idx] == step) >= 1

            # Attr. activation
            # print('Model act codes: '+str(mas[q_ind]))
            avl_mods = np.array(mas[q_idx]) > 0  # Update required

            # print('Avl mods: '+str(avl_mods))
            avl_mod_codes = m_codes[avl_mods]

            att_appr_scores = [np.sum(avl_mod_codes[:, a_ind] == 1)
                               if (avl_atts[a_ind] == 0) else -1
                               for a_ind in range(nb_atts)]  # Only score non-available ones!
            # print('att_appr_scores: '+str(att_appr_scores))
            aas[q_idx] = [step if (att_appr_scores[a_ind] > 0) else v
                                    for a_ind, v
                                    in enumerate(aas[q_idx])]
            aas[q_idx] = np.array(aas[q_idx])

            # Loop technics
            att_progress = np.sum(aas[q_idx] == step)
            if ((att_progress > 0) & (step < max_layers)):
                step += 1
                thresholds = iter(np.arange(init_threshold, -1, -step_size))  # Reset thresholds

            done = (np.sum(aas[q_idx][q_targ[q_idx]] < 0) == 0)

    return np.array(mas), np.array(aas)


def rw_pred_algo(m_codes, q_codes, settings):
    """
    Random Walk prediction algo.

    Generate one random walk in the random forest.

    :param m_codes:
    :param q_codes:
    :param settings:
    :return:
    """

    # Preliminaries
    nb_models, nb_atts, nb_queries, m_desc, m_targ, q_desc, q_targ = pred_prelims(m_codes,
                                                                                  q_codes)
    mas, aas = init_mas_aas(nb_models, nb_atts, nb_queries)

    # Building codes
    for q_idx, q_code in enumerate(q_codes):
        mas[q_idx], aas[q_idx] = generate_chain(m_codes,
                                                q_desc[q_idx],
                                                q_targ[q_idx],
                                                settings)

    return np.array(mas), np.array(aas)


def generate_chain(m_codes, q_desc, q_targ, settings):
    # Choose chain length
    assert isinstance(settings['its'], int)
    FI = settings['FI']
    max_size = settings['its'] + 1
    chain_size = np.random.randint(1, max_size)
    steps = np.arange(chain_size, 0, -1, dtype=int)

    # Initializations
    nb_models = len(m_codes)
    nb_atts = len(m_codes[0])

    mas = np.full(nb_models, 0, dtype=np.int)
    aas = np.full(nb_atts, -1, dtype=np.int)
    aas[q_desc] = 0

    for i, step in enumerate(steps):
        # Select possible targets
        if i == 0:
            potential_targ = q_targ
        else:
            assert step >= 1
            next_mods = (mas == step + 1)
            potential_targ = get_atts(m_codes, next_mods, 0)

        potential_targ = np.array([e for e in potential_targ
                                   if aas[e] == -1])  # Predict only unknown atts
        if len(potential_targ) == 0: break  # Nothing left to contribute

        # Select potential models (= model which predicts a potential target)
        potential_mods = np.array([m_idx for m_idx, e in enumerate(mas)
                                   if (e == 0)
                                   if (np.sum(m_codes[m_idx, potential_targ] == 1) > 0)])

        # NEW ADDITION I DONT KNOW IF THIS IS OKAY
        if len(potential_mods) == 0: break  # Nothing left to contribute

        # Select model
        available_atts = [1 if (-1 < aas[i] < step) else 0 for i in range(len(aas))]
        mod_appr_scores = [np.dot(available_atts, FI[m_idx]) for m_idx in potential_mods]

        curr_mod_idx = potential_mods[pick_random_models_from_appr_score(mod_appr_scores, n=1) > 0]

        mas[curr_mod_idx] = step

        # Select target
        curr_mods_targ = get_atts(m_codes, curr_mod_idx, 1)
        relv_targ_idx = np.intersect1d(potential_targ, curr_mods_targ)

        aas[relv_targ_idx] = step


    if not np.max(mas) > 0:
        print("i, step: {},{}".format(i,step))
        print("potential targ: {}".format(potential_targ))
        print("aas: {}".format(aas))
        print("m_codes:\n {}".format(m_codes))
        print(type(m_codes))
        print("steps {}".format(steps))
        print("chain_size {}".format(chain_size))
        print("q_desc {}".format(q_desc))
        print("q_targ {}".format(q_targ))

    assert np.max(aas) == np.max(mas)
    assert np.max(mas) > 0
    mas, aas = recode_strat(mas, aas)

    return mas, aas


## Utillities
def pred_prelims(m_codes, q_codes):
    """
    Some things that every prediction strategy needs.

    :param m_codes:
    :param q_codes:
    :return:
    """

    nb_models, nb_atts, nb_queries = len(m_codes), len(m_codes[0]), len(q_codes)
    m_desc, m_targ, _ = codes_to_query(m_codes)
    q_desc, q_targ, _ = codes_to_query(q_codes)

    return nb_models, nb_atts, nb_queries, m_desc, m_targ, q_desc, q_targ


def init_mas_aas(nb_models, nb_atts, nb_queries):
    """
    Initialization of the prediction strategies.

    Shared between different prediction strategies.

    :param nb_models:
    :param nb_atts:
    :param nb_queries:
    :return:
    """

    mas = [np.zeros((nb_models), dtype=np.int) for j in range(nb_queries)]
    aas = [np.full((nb_atts), -1, dtype=np.int) for j in range(nb_queries)]
    return mas, aas


def prune_strat(m_codes, q_code, mas, aas):
    """
    Prune the last step of the activation strategies.

    :param m_codes:         model codes
    :param q_code:          code of the queries
    :param mas:             Model Activation Strategy
    :param aas:             Attribute Activation Strategy
    :return:
    """

    step = np.max(aas)
    aas = np.array([e if ((e != step) | (q_code[i] == 1)) else 0
                    for i, e in enumerate(aas)])

    act_atts = (aas == step)
    mas = [e if ((e != step) | (np.sum(m_codes[i][act_atts] == 1) > 0))
           else 0
           for i, e in enumerate(mas)]

    return np.array(mas), np.array(aas)


def full_prune_strat(m_codes, q_code, mas, aas):
    """
    Prune the last step of the activation strategies.

    :param m_codes:         model codes
    :param q_code:          code of the queries
    :param mas:             Model Activation Strategy
    :param aas:             Attribute Activation Strategy
    :return:
    """

    max_step = np.max(aas)

    for i in range(max_step):
        step = max_step - i

        if step == max_step:
            aas = np.array([e if ((e != step) | (q_code[i] == 1))
                            else 0 for i, e in enumerate(aas)])
        else:
            next_act_mods = (mas > step)
            # print(next_act_mods)
            aas = np.array(
                [e if ((e != step) | (np.sum(m_codes[next_act_mods, i] == 0) > 0) | (q_code[i] == 1))
                 else 0 for i, e in enumerate(aas)])

        act_atts = (aas >= step)
        mas = [e if ((e != step) | (np.sum(m_codes[i][act_atts] == 1) > 0))
               else 0 for i, e in enumerate(mas)]

    # Recode strategies
    mas, aas = recode_strat(mas, aas)

    return mas, aas


def recode_strat(mas, aas):
    """
    Avoid gaps in strategy, and make sure that it starts from one.

    :param mas:     Model Activation Strategy
    :param aas:     Attribute Activation Strategy
    :return:
    """

    # Recode the arrays, some layers might disappear completely!
    uniq_mas, uniq_aas = np.unique(mas), np.unique(aas)
    uniq_mas = np.array([e for e in uniq_mas if e > 0])
    uniq_aas = np.array([e for e in uniq_aas if e > 0])
    mas = np.array([np.where(uniq_mas == e)[0][0] + 1 if e > 0 else e for e in mas])
    aas = np.array([np.where(uniq_aas == e)[0][0] + 1 if e > 0 else e for e in aas])  # -1 for unknown targets

    return mas, aas


def get_atts(m_codes, m_idx, role=0):
    """
    Get the attributes that fulfill a certain role in the selected models.

    The role refers to the m_codes, i.e. 0 = desc, -1 = missing, 1 = target.
    """

    all_atts = np.where(m_codes[m_idx, :] == role)[1]  # Where all the attributes with 'role' are
    unique_atts = np.unique(all_atts)
    return unique_atts


def pick_random_models_from_appr_score(mod_appr_scores, n=1):
    norm = np.linalg.norm(mod_appr_scores, 1)
    if norm > 0:
        mod_appr_scores = mod_appr_scores/norm
    else:
        # If you cannot be right, be arbritrary
        mod_appr_scores = [1/len(mod_appr_scores) for i in mod_appr_scores]

    return np.random.multinomial(n, mod_appr_scores, size=1)[0]
