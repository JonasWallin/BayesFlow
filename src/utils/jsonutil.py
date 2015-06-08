import json
import numpy as np
import inspect
#from ..HMlog import HMlogB,HMlog

'''
    Encoding
'''

class ArrayEncoder(json.JSONEncoder):
    def default(self, o):
        #print "decoding {} with type {}".format(o,type(o))
        if isinstance(o,np.ndarray):
            return {'shape':o.shape,'data':o.tolist(),'__type__':'np.ndarray','dtype':str(o.dtype)}
        if isinstance(o,np.int32):
            return int(o)
        return super(ArrayEncoder,self).default(o)

class ObjJsonEncoder(ArrayEncoder):
    def default(self,o):
        if hasattr(o,'encode_json'):
            return(o.encode_json())
        return super(ObjJsonEncoder,self).default(o)

# class ArrayEncoder(json.JSONEncoder):
#     def default(self, o):
#         if isinstance(o,np.ndarray):
#             return {'shape':o.shape,'data':list(o.ravel()),'__type__':'np.ndarray'}
#         return super(ArrayEncoder,self).default(o)

# class HMlogBEncoder(ArrayEncoder):
#      def default(self, o):
#         if isinstance(o,HMlogB):
#             jsondict  = {'__type__':'HMlogB'}
#             for arg in ['savefrq','nbrsave','sim','K','d','noise_class','names',
#                         'active_komp','lab_sw','theta_sim','nu_sim']:
#                 jsondict.update({arg:getattr(o,arg)})             
#             return jsondict            
#         return super(HMlogBEncoder,self).default(o)

# class HMlogEncoder(HMlogBEncoder):
#     def default(self,o):
#         if isinstance(o,HMlog):
#             jsondict = super(HMlogEncoder,self).default(o)
#             for arg in ['theta_sim_mean','Sigmaexp_sim_mean','mupers_sim_mean','Sigmapers_sim_mean',
#                         'prob_sim_mean','J']:
#             jsondict.update(arg:getattr(o,arg))
#         return super(HMlogEncoder,self).default(o)

'''
    Decoding
'''

def class_decoder(obj,Cls):
    if not '__type__'in obj or obj['__type__'] == 'np.ndarray':
        return array_decoder(obj)
    classname = obj['__type__']
    del obj['__type__']
    obj_decode = construct_from_dict(obj,Cls)
    for arg in obj:
        setattr(obj_decode,arg,obj[arg])
    return obj_decode

def array_decoder(obj):
    if '__type__' in obj and obj['__type__'] == 'np.ndarray':
        return np.asarray(obj['data'],dtype=obj['dtype'])
    return obj

def construct_from_dict(dic,Cls):
    init_args,_,_,defaults = inspect.getargspec(Cls.__init__)
    print "init_args = {}".format(init_args)
    init_dict = {}
    for i in range(1,len(init_args)+1):
        arg = init_args[-i]
        try:
            init_dict[arg] = dic[arg]
            del dic[arg]
        except:
            try:
                init_dict[arg] = defaults[-i]
            except:
                if arg == 'self':
                    continue
                if arg == 'hGMM':
                    init_dict[arg] = construct_from_dict(dic,hier_mixture_mpi_mimic)
                raise KeyError, 'Attribute needed for constructor not provided'
    return Cls(**init_dict)


class GMM_mimic(object):
    def __init__(self,name):
        self.name = name

class hier_mixture_mpi_mimic(object):
    def __init__(self,K,d,names,noise_class):
        self.K = K
        self.d = d
        self.GMMs = [GMM_mimic(name) for name in names]
        self.noise_class = noise_class

# def array_decoder(obj):
#     if '__type__' in obj and obj['__type__'] == 'np.ndarray':
#         return np.array(obj['data']).reshape(obj['shape'])
#     return obj

# def hmlogb_decoder(obj):
#     if '__type__' in obj and obj['__type__'] == 'HMlogB':
#         hGMM = hier_mixture_mpi_mimic(obj['K'],obj['d'],obj['names'],obj['noise_class'])
#         hmlogb = HMlogB(hGMM,obj['sim'],obj['nbrsave'],obj['savefrq'])
#         for arg in ['active_komp','lab_sw','theta_sim','nu_sim']:
#             setattr(hmlogb,arg) = obj[arg]
#         return hmlogb
#     return array_decoder(obj)

# def hmlog_decoder(obj):
#     if '__type__' in obj and obj['__type__'] == 'HMlog':
#         hmlog = hmlogb_decoder(obj)
#         for arg in ['theta_sim_mean','Sigmaexp_sim_mean','mupers_sim_mean','Sigmapers_sim_mean',
#                     'prob_sim_mean','J']:
#             setattr(hmlog,arg) = obj['arg']
#         return hmlog
#     return hmlogb_decoder(obj)



