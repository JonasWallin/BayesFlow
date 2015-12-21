import json
import numpy as np
import inspect

'''
    Encoding
'''


class ArrayEncoder(json.JSONEncoder):
    def default(self, o):
        #print "decoding {} with type {}".format(o, type(o))
        if isinstance(o, np.ndarray):
            return {'shape': o.shape, 'data': o.tolist(), '__type__': 'np.ndarray', 'dtype': str(o.dtype)}
        if isinstance(o, np.int32):
            return int(o)
        return super(ArrayEncoder, self).default(o)


class ObjJsonEncoder(ArrayEncoder):
    def default(self, o):
        if hasattr(o, 'encode_json'):
            return(o.encode_json())
        return super(ObjJsonEncoder, self).default(o)

'''
    Decoding
'''


def class_decoder(obj, Cls, **kwargs):
    '''
        Cls can be a dictionary of classes or a single class.
    '''
    if not '__type__'in obj or obj['__type__'] == 'np.ndarray':
        return array_decoder(obj)
    #print "type = {}".format(obj['__type__'])
    try:
        objCls = Cls[obj['__type__']]
    except TypeError:
        objCls = Cls
    del obj['__type__']
    #print "obj.keys() = {}".format(obj.keys())
    obj_decode = construct_from_dict(obj, objCls, **kwargs)
    #print "{} constructed from dict".format(objCls)
    for arg in obj:
        setattr(obj_decode, arg, obj[arg])
    return obj_decode


def array_decoder(obj):
    if '__type__' in obj and obj['__type__'] == 'np.ndarray':
        return np.asarray(obj['data'], dtype=obj['dtype'])
    return obj


def construct_from_dict(dic, Cls, **kwargs):
    init_args, _, _, defaults = inspect.getargspec(Cls.__init__)
    #print "Cls = {}".format(Cls)
    #print "init_args = {}".format(init_args)
    init_dict = {}
    for i in range(1, len(init_args)+1):
        arg = init_args[-i]
        try:
            init_dict[arg] = dic[arg]
            del dic[arg]
        except KeyError:
            try:
                init_dict[arg] = kwargs[arg]
            except KeyError:
                try:
                    init_dict[arg] = defaults[-i]
                except:
                    if arg == 'self':
                        continue
                    if arg == 'hGMM':
                        init_dict[arg] = construct_from_dict(dic, hier_mixture_mpi_mimic)
                        continue
                    raise KeyError('Attribute {} needed for constructor not provided'.format(arg))
    return Cls(**init_dict)


class GMM_mimic(object):
    def __init__(self, name, noise_mu=None, noise_sigma=None):
        self.name = name
        self.data = np.empty(0)
        self.noise_mean = noise_mu
        self.noise_sigma = noise_sigma


class hier_mixture_mpi_mimic(object):
    def __init__(self, K, d, names, noise_class, noise_mu=None, noise_sigma=None):
        self.K = K
        self.d = d
        self.GMMs = [GMM_mimic(name, noise_mu, noise_sigma) for name in names]
        self.noise_class = noise_class

if __name__ == '__main__':
    class Foo(object):
        def __init__(self, a, b=1, bar=None):
            self.a = a
            self.b = b
            self.bar = bar

        def encode_json(self):
            dic = {"__type__": "Foo"}
            for arg in self.__dict__:
                dic[arg] = getattr(self, arg)
            return dic

        @classmethod
        def load(cls, json_str):
            return json.loads(foo_json, object_hook=lambda obj:
                              class_decoder(obj, {'Foo': cls, 'Bar': Bar}))

    class Bar(object):
        def __init__(self):
            self.s = 8

        def encode_json(self):
            return {'__type__': 'Bar', 's': self.s}

    foo = Foo(5, bar=Bar())
    foo_json = json.dumps(foo, cls=ObjJsonEncoder)
    print "foo_json = {}".format(foo_json)
    foo_load = Foo.load(foo_json)
    print "foo_load = {}".format(foo_load.__dict__)
    print "foo_load.bar.__dict__ = {}".format(foo_load.bar.__dict__)
