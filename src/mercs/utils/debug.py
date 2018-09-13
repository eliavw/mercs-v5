
def debug_print(msg, level=1, V=0, **kwargs):

    """
    def get_var_name(var):
        var_name = [k for k, v in locals().items() if v is var][0]
        return var_name

    if kwargs.keys() > {'msg', 'level', 'V'}:
        print('INSIDE')
        relevant = {k:v for k,v in kwargs.items()
                    if k not in {'msg', 'level', 'V'}}
        for k,v in relevant.items():
            msg+="k: {}".format(v)
    """

    if V >= level:
        print(msg+"\n")
    return