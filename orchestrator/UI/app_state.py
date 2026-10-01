class AppState(dict):
    """Модифицированный словарь.

    К атрибутам можно обращаться как через object.attr, так и через object['attr'].

    Example:
        m = AppState({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])

        f_name = m.first_name
        f_name = m['first_name'] # То же самое
    """

    def __init__(self, *args, **kwargs):
        super(AppState, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(AppState, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(AppState, self).__delitem__(key)
        del self.__dict__[key]
