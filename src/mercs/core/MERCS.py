import json

from sklearn.preprocessing import Imputer
from timeit import default_timer

from ..algo.induction import base_ind_algo
from ..algo.prediction import *
from ..algo.selection import *
from ..io.io import save_output_data
from ..models.PolyModel import *
from ..settings import *
from ..utils.keywords import *
from ..utils.metadata import get_metadata_df

from ..utils.debug import debug_print
VERBOSITY = 1


class MERCS(object):
    """
    A MERCS model

        |
        |
        v
        Multi-directional
        Ensembles of
        Regression and
        Classification
    treeS
    """

    # Main Methods
    def __init__(self, settings_fname=None):
        """
        Return an initialized MERCS Classifier object.

        :param settings_fname:    Filename of .json file containing settings
        """

        if settings_fname is None:
            self.s = create_settings()
        else:
            # TODO(elia): Merge what I load with default dict
            self.load_settings(settings_fname)

        self.m_codes = None
        self.m_list = None
        self.q_models = None
        self.imputator = None

    def fit(self, X, **kwargs):
        """
        Fit the MERCS model to a dataset X.

        This happens in 2 steps:
        1. Selection
            This is choosing which models will be part of our MERCS ensemble.
        2. Actual Induction
            This is training the selected models.

        :param X:           Pandas DataFrame, training set
        :param **kwargs:    Keyword arguments that can modify settings.
        :return:
        """

        # 0. Prelims
        tick = default_timer()
        self.s['metadata'] = get_metadata_df(X)

        msg="""
        metadata of our model is: {}
        """.format(self.s['metadata'])
        debug_print(msg,V=VERBOSITY)

        self.update_settings(mode='fit', **kwargs)
        self.fit_imputator(X)

        # 1. Selection = Prepare Induction
        self.m_codes = self.perform_selection(self.s['metadata'])

        # 2. Induction
        self.m_codes, self.m_list = self.perform_induction(X,
                                                           self.m_codes,
                                                           self.s['induction'],
                                                           self.s['metadata'])
        # 3. Post processing
        tock = default_timer()
        self.update_settings(mode='metadata')   # Save info on learned models
        self.update_settings(mode='model_data', mod_ind_time=tock-tick)

        return

    def predict(self, X, q_idx=0, **kwargs):
        """
        Predict Y from X.

        Which attributes to predict are specified by queries.

        :param q_idx:       Optional keyword identifying which of the already
                            loaded queries you want to predict.
        :param X:           Test dataset. (Pandas DataFrame)
        :param kwargs:      Optional keyword arguments.
                            Used in updating the settings of the MERCSClassifier.
        :return:
        """

        tick = default_timer()
        # 0. Settings
        self.update_settings(mode='predict', **kwargs)

        # 1. Prediction = Prepare Inference
        self.q_models = self.query_to_model(self.m_list,
                                            self.m_codes,
                                            self.s['prediction'],
                                            self.s['metadata'],
                                            q_code=self.s['queries']['codes'][q_idx])

        print("Predicting q_code: {}".format(self.s['queries']['codes'][q_idx]))
        # 2. Inference
        X_query = perform_imputation(X,
                                     self.s['queries']['codes'][q_idx],
                                     self.imputator)  # Generate X data_csv.

        Y = self.q_models[q_idx].predict(X_query)
        del X_query


        tock = default_timer()
        self.update_settings(mode='model_data', mod_inf_time=tock-tick)

        del self.q_models

        return Y

    def predict_proba(self, X, q_idx=0, **kwargs):
        """
        Predict Y from X_test.

        Which attributes to predict are specified by queries.

        :param q_idx:       Optional keyword identifying which of the already
                            loaded queries you want to predict.
        :param X:           Test dataset. (Pandas DataFrame)
        :param kwargs:      Optional keyword arguments.
                            Used in updating the settings of the MERCSClassifier.
        :return:
        """

        tick = default_timer()
        # 0. Settings
        self.update_settings(mode='predict', **kwargs)

        # 1. Prediction = Prepare Inference
        self.q_models = self.query_to_model(self.m_list,
                                            self.m_codes,
                                            self.s['prediction'],
                                            self.s['metadata'],
                                            q_code=self.s['queries']['codes'][q_idx])
        # 2. Inference
        X_query = perform_imputation(X,
                                     self.s['queries']['codes'][q_idx],
                                     self.imputator)  # Generate X data_csv.

        Y_proba = self.q_models[q_idx].predict_proba(X_query)
        del X_query

        tock = default_timer()
        self.update_settings(mode='model_data', mod_inf_time=tock-tick)

        return Y_proba

    def batch_predict(self, X, fnames, **kwargs):
        tick = default_timer()
        # 0. Settings
        self.update_settings(mode='batch_predict', **kwargs)

        nb_queries = len(self.s['queries']['codes'])
        assert nb_queries == len(fnames)

        # 1. Prediction = Prepare Inference
        self.q_models = self.query_to_model(self.m_list,
                                            self.m_codes,
                                            self.s['prediction'],
                                            self.s['metadata'],
                                            q_codes=self.s['queries']['codes'])

        # 2. Inference
        for q_idx in range(nb_queries):
            # Generate X data_csv for queries with index q_idx
            X_query = perform_imputation(X,
                                         self.s['queries']['codes'][q_idx],
                                         self.imputator)

            Y = self.q_models[q_idx].predict(X_query)

            del X_query

            save_output_data(Y,
                             self.s['queries']['q_targ'][q_idx],
                             fnames[q_idx])
            del Y

        tock = default_timer()
        self.update_settings(mode='model_data', mod_inf_time=tock-tick)

        return

    # 0. Preliminaries
    def load_settings(self, filename, mode=None):
        """
        Load a JSON settingsfile.

        The settings are saved and loaded in JSON (dict) format.
        """

        with open(filename) as f:
            new_settings = json.load(f)

        self.import_settings(new_settings, mode=mode)

        return

    def import_settings(self, new_settings, mode=None):
        """
        Import an external settings dictionary into the classifier.



        :param new_settings:
        :param mode:
        :return:
        """

        if mode in {'induction','ind'}:
            self.s['induction'] = new_settings
        elif mode in {'selection','sel'}:
            self.s['selection'] = new_settings
        elif mode in {'prediction', 'pred'}:
            self.s['prediction'] = new_settings
        elif mode in {'queries', 'queries', 'q', 'qry'}:
            self.s['queries'] = new_settings
        elif mode in {'metadata', 'md'}:
            self.s['metadata'] = new_settings
        elif mode in {'model_data', 'mod'}:
            self.s['model_data'] = new_settings
        elif mode in {'algo', 'do', 'main'}:
            self.import_settings(new_settings['induction'], mode='induction')
            self.import_settings(new_settings['selection'], mode='selection')
            self.import_settings(new_settings['prediction'], mode='prediction')
        elif mode == 'fit':
            # Assuming that new_settings has keys induction and selection
            self.import_settings(new_settings['induction'], mode='induction')
            self.import_settings(new_settings['selection'], mode='selection')
        elif mode in {'predict','batch_predict'}:
            # Assuming that new_settings has key prediction
            self.import_settings(new_settings['prediction'], mode='prediction')
        else:
            # If no mode provided, assume global settings
            warnings.warn("Did not recognize mode: {}."
                          "Assuming algorithm settings.".format(mode))
            self.s = new_settings

        return

    def update_settings(self, mode=None, delimiter='_', **kwargs):
        """
        Update the settings dictionary.

        :param mode:            Settings category that has to be updated
        :param delimiter:       Delimiter of the settings keywords
        :param kwargs:          Keyword arguments
        :return:
        """

        if mode in {'induction','ind'}:
            self.s['induction'] = filter_kwargs_update_settings(self.s['induction'],
                                                                prefix='ind',
                                                                delimiter=delimiter,
                                                                **kwargs)
        elif mode in {'selection','sel'}:
            self.s['selection'] = filter_kwargs_update_settings(self.s['selection'],
                                                                prefix='sel',
                                                                delimiter=delimiter,
                                                                **kwargs)
        elif mode in {'prediction','pred'}:
            self.s['prediction'] = filter_kwargs_update_settings(self.s['prediction'],
                                                                 prefix='pred',
                                                                 delimiter=delimiter,
                                                                 **kwargs)
        elif mode in {'queries','queries', 'q', 'qry'}:
            nb_atts = self.s['metadata'].get('nb_atts', 0)
            if nb_atts > 1:
                self.s['queries'] = update_query_settings(self.s['queries'],
                                                          nb_atts,
                                                          delimiter=delimiter,
                                                          **kwargs)
        elif mode in {'metadata','md'}:
            self.s['metadata'] = update_meta_data(self.s['metadata'],
                                                  self.m_list,
                                                  self.m_codes)
        elif mode in {'model_data'}:
            self.s['model_data'] = filter_kwargs_update_settings(self.s['model_data'],
                                                                 prefix='mod',
                                                                 delimiter=delimiter,
                                                                 **kwargs)
        elif mode in {'fit'}:
            self.update_settings(mode='induction', delimiter=delimiter, **kwargs)
            self.update_settings(mode='selection', delimiter=delimiter, **kwargs)
        elif mode in {'predict','batch_predict'}:
            self.update_settings(mode='prediction', delimiter=delimiter, **kwargs)
            self.update_settings(mode='queries', delimiter=delimiter, **kwargs)
        else:
            warnings.warn("Did not recognize mode: {}. "
                          "Updating all settings.".format(mode))
            self.update_settings(mode='induction', delimiter=delimiter, **kwargs)
            self.update_settings(mode='selection', delimiter=delimiter, **kwargs)
            self.update_settings(mode='prediction', delimiter=delimiter, **kwargs)
            self.update_settings(mode='queries', delimiter=delimiter, **kwargs)
            self.update_settings(mode='metadata', delimiter=delimiter, **kwargs)
            self.update_settings(mode='model_data', delimiter=delimiter, **kwargs)

        return

    def fit_imputator(self, X):
        """
        Construct and fit an imputator based on input data_csv.

        This to fill in missing values later on.
        """
        imputator = Imputer(missing_values='NaN',
                            strategy='most_frequent',
                            axis=0)
        imputator.fit(X)

        self.imputator = imputator

        return

    # 1. Selection = Prepare Induction
    def perform_selection(self, metadata):
        """
        Generate m_codes by a selection algorithm

        :param metadata:    Dict with necessary info for selection algo
        :return:
        """

        sel_type = self.s['selection']['type']
        keywords = kw_sel_type()

        if sel_type in keywords['base']:
            m_codes = base_selection_algo(metadata,
                                          self.s['selection'])
        elif sel_type in keywords['random']:
            m_codes = random_selection_algo(metadata,
                                            self.s['selection'])
        else:
            warnings.warn("Did not recognize selection algorithm {}"
                          "Available algorithms are {}".format(sel_type, keywords.keys()))
            m_codes = base_selection_algo(metadata,
                                          self.s['selection'],
                                          target_atts_list=metadata['att_types']['nominal'])

        return m_codes

    # 2. Perform Induction
    def perform_induction(self, df, m_codes, settings, metadata):
        """
        Actual induction of the mde model.

        Returns a list of sklearn classifiers.
        """

        m_desc, m_targ, _ = codes_to_query(m_codes)

        # Build m_list (unfitted)
        m_list = base_ind_algo(metadata, settings, m_targ)
        nb_models = len(m_list)

        # Fit all the component models
        for i in range(nb_models):
            assert isinstance(m_desc[i], list)
            assert isinstance(m_targ[i], list)

            m_atts = m_desc[i] + m_targ[i]

            assert len(m_atts) == len(m_desc[i])+len(m_targ[i])

            X_Y = df.iloc[:, m_atts].dropna().values
            X = X_Y[:, :len(m_desc[i])]
            Y = X_Y[:, len(m_targ[i]):]

            msg="""
            X.shape: {}\n
            Y.shape: {}\n
            """.format(X.shape, Y.shape)
            debug_print(msg, V=VERBOSITY, warn=True)

            # Convert (m X 1)-dim arrays to (m, )-dim arrays
            if 1 in list(X.shape): X = X.ravel()
            if 1 in list(Y.shape): Y = Y.ravel()

            m_list[i].fit(X, Y)
            del X, Y, X_Y

        flatten = self.s['induction'].get('flatten', False)
        if flatten:
            m_list, m_codes = self.flatten_model(m_list, m_codes)

        return m_codes, m_list

    # 3. Prediction = Prepare Inference
    def query_to_model(self, m_list, m_codes, settings, metadata, **kwargs):
        """
        Convert a given queries to a model that answers exactly that queries.

            1. Convert the queries to a Model Activation Strategy (MAS) and a Attribute Activation Strategy (AAS)
            2. Convert the MAS, AAS to a model

        :return:

        TODO(elia): Optimize this. Both the prediction functions and the model builder to many things the same time, e.g.:
                1. Converting queries codes and model codes
                2. Nb_atts, Nb_queries, etc. This is all in the metadata!
        """

        qry_keywords = kw_qry_codes()

        # Prelims
        # TODO: Make prediction methods handle settings more elegantly
        new_settings = {**settings,
                        'clf_labels':   metadata['clf_labels'],
                        'FI':           metadata['FI']}

        for k in kwargs:
            if k in qry_keywords['query_codes']:            # Multiple codes are OK
                q_codes = kwargs[k]
            elif k in qry_keywords['query_code']:           # Single code in array
                q_codes = [kwargs[k]]
            else:
                raise ValueError("Did not recognize keyword."
                                 "Allowed qry_keywords: {}".format(qry_keywords))

        # Actual work
        if new_settings['type'] == 'MI':
            mas, aas = mi_pred_algo(m_codes, q_codes)
            query_models = self.strat_to_model(m_list,
                                               m_codes,
                                               q_codes,
                                               mas,
                                               aas,
                                               metadata)
        elif new_settings['type'] == 'MA':
            mas, aas = ma_pred_algo(m_codes, q_codes, new_settings)
            query_models = self.strat_to_model(m_list,
                                               m_codes,
                                               q_codes,
                                               mas,
                                               aas,
                                               metadata)
        elif new_settings['type'] == 'MAFI':
            mas, aas = mafi_pred_algo(m_codes, q_codes, new_settings)
            query_models = self.strat_to_model(m_list,
                                               m_codes,
                                               q_codes,
                                               mas,
                                               aas,
                                               metadata)
        elif new_settings['type'] == 'IT':
            mas, aas = it_pred_algo(m_codes, q_codes, new_settings)
            for i, query_code in enumerate(q_codes):
                mas[i], aas[i] = full_prune_strat(m_codes, q_codes[i], mas[i], aas[i]) #TODO(elia) This should not happen here
            query_models = self.strat_to_model(m_list,
                                               m_codes,
                                               q_codes,
                                               mas,
                                               aas,
                                               metadata)
        elif new_settings['type'] == 'RW':
            assert isinstance(new_settings['param'], int)
            nb_walks = new_settings['param']
            all_q_mods = [None] * nb_walks

            for rw_idx in range(nb_walks):
                mas, aas = rw_pred_algo(m_codes, q_codes, new_settings)

                all_q_mods[rw_idx] = self.strat_to_model(m_list,
                                                         m_codes,
                                                         q_codes,
                                                         mas,
                                                         aas,
                                                         metadata)

            all_q_mods = np.array(all_q_mods)
            _, q_targ, _ = codes_to_query(q_codes)
            query_models = [build_ensemble_model(all_q_mods[:, i], targ, metadata)
                            for i, targ in enumerate(q_targ)]

        else:
            warnings.warn("\nDid not recognize prediction method: '{}'\n"
                          "Using MI algorithm instead".format(new_settings['type']))

            mas, aas = mi_pred_algo(m_codes, q_codes)
            query_models = self.strat_to_model(m_list,
                                               m_codes,
                                               q_codes,
                                               mas,
                                               aas,
                                               metadata)

        return query_models

    def strat_to_model(self, m_list, m_codes, q_codes, mas, aas, metadata):
        """
        Convert the MAS and AAS to a single, grouped model.

        The use of this is that the model and attribute activation strategy gets 'locked in'
        a special purpose model.

        :param m_list:          List of models
        :param m_codes:         Characteristic code of the models
        :param mas:             Model Activation Strategy
        :param aas:             Attribute Activation Strategy
        :param q_codes:         Characteristic code of the queries this model was built to address
        :param metadata:        Metadata of the composing models
        :return:
        """

        # Prelims
        assert len(mas) == len(aas) == len(q_codes)
        m_desc, m_targ, _ = codes_to_query(m_codes)
        _, q_targ, _ = codes_to_query(q_codes)

        # Convert every MAS-AAS combination to a dedicated model
        query_models = [build_chained_model(m_list, m_desc, m_targ, q_targ[i], mas[i], aas[i], metadata)
                        for i in range(len(q_codes))]

        return np.array(query_models)

    # 4. Advanced functionalities
    def merge(self, other):
        """
        Merge current mercs model with another one.

        :param other:   Other mercs model
        :return:
        """

        own_codes = self.m_codes
        new_codes = other.m_codes
        self.m_codes = np.concatenate((own_codes, new_codes))
        self.m_list.extend(other.m_list)

        self.q_models = None # Just a reset
        self.update_settings(mode='metadata') # Save info on learned models

        return

    def flatten_model(self, m_list, m_codes):
        """
        Method to unravel composite models (i.e. RandomForests) to its
        fundamental components (i.e. DecisionTrees)

        :return:
        """

        if isinstance(m_list[0], (RandomForestClassifier, RandomForestRegressor)):

            # Actual actions
            new_m_list = []
            new_m_codes = m_codes[0:1]

            for i, m in enumerate(m_list):
                new_m_list.extend(m)
                codes = np.tile(m_codes[i], (len(m), 1))
                new_m_codes = np.concatenate((new_m_codes, codes))

            new_m_codes = new_m_codes[1:]  # First line was filled in as an initialization and has to be gone.

            return new_m_list, new_m_codes

        else:
            return m_list, m_codes
