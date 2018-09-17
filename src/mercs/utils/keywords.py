"""
In this file we define functions that return keywords for certain contexts

In this way, we try to enforce system-wide consistency for a better
user experience.
"""


def kw_ind_trees():
    keywords = {'dt',
                'DT',
                'tree',
                'Tree',
                'dtree',
                'Dtree',
                'DTree',
                'decisiontree',
                'Decisiontree',
                'DecisionTree'}
    return keywords


def kw_ind_forests():
    keywords = {'rf',
                'RF',
                'Rforest',
                'RForest',
                'forest',
                'Forest',
                'randomforest',
                'RandomForest'}
    return keywords


def kw_sel_type():

    keywords = {'base':     {'base',
                             'Base'},
                'random':   {'rnd',
                             'random',
                             'Random'}}

    return keywords


def kw_qry_codes():
    keywords = {'query_code':   {'q_code',
                                 'qry_code',
                                 'queries',
                                 'query_code'},

                'query_codes':  {'q_codes',
                                 'qry_codes',
                                 'queries',
                                 'query_codes'}}
    return keywords
