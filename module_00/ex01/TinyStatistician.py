import numpy as np

def interpolation(d,x):
            output=d[0][1]+(x-d[0][0])*((d[1][1]-d[0][1])/(d[1][0]-d[0][0]))
            return output
class TinyStatistician:
    @staticmethod
    def mean(x):
        if not isinstance(x,(list,np.ndarray)):
            print('argument has to be a list or a numpy array.')
            return None
        err_msg='argument must not contain non-numeric types.'
        if isinstance(x,np.ndarray) and not np.issubdtype(x.dtype,np.number):
            print(err_msg)
            return None
        elif isinstance(x,list):
            types=set(map(type,x))
            if len(types)>(int in types)+(float in types):
                print(err_msg)
                return None
        u=0
        for elem in x:
            u+=elem
        return float(u/len(x))
    @staticmethod
    # numpy percentiles using linear interpolation between the two closest list index
    def np_percentile(x,p):
        if not isinstance(x,(list,np.ndarray)):
            print('argument has to be a list or a numpy array.')
            return None
        err_msg='argument must not contain non-numeric types.'
        if isinstance(x,np.ndarray) and not np.issubdtype(x.dtype,np.number):
            print(err_msg)
            return None
        elif isinstance(x,list):
            types=set(map(type,x))
            if len(types)>(int in types)+(float in types):
                print(err_msg)
                return None
        if not isinstance(p,int) or p<0 or p>99:
            print('p has to be an int between 0 and 99.')
            return None
        x=sorted(x)
        q_value=len(x)*p/100
        if q_value.is_integer():
            if q_value in [i for i,_ in enumerate(x)]:
                return float(x[int(q_value)])
        q_value=int(q_value)
        data=[[q_value,x[q_value]],[q_value+1,x[q_value+1]]]
        return(float(interpolation(data,p*len(x)/100)))
    @staticmethod
    # population basic percentiles
    def percentile(x,p):
        if not isinstance(x,(list,np.ndarray)):
            print('argument has to be a list or a numpy array.')
            return None
        err_msg='argument must not contain non-numeric types.'
        if isinstance(x,np.ndarray) and not np.issubdtype(x.dtype,np.number):
            print(err_msg)
            return None
        elif isinstance(x,list):
            types=set(map(type,x))
            if len(types)>(int in types)+(float in types):
                print(err_msg)
                return None
        if not isinstance(p,int) or p<0 or p>99:
            print('p has to be an int between 0 and 99.')
            return None
        x=sorted(x)
        idx=p*len(x)//100
        return float(x[idx])
    def median(x):
        return TinyStatistician.percentile(x,50)
    @staticmethod
    def quartile(x):
        res25=TinyStatistician.percentile(x,25)
        if res25 is None:
            return None
        res75=TinyStatistician.percentile(x,75)
        if res75 is None:
            return None
        return [res25,res75]
    @staticmethod
    def var(x):
        m=TinyStatistician.mean(x)
        if m is None:
            return None
        v=0
        for elem in x:
            v+=(elem-m)**2
        return float(v/(len(x)-1))
    @staticmethod
    def std(x):
        v=TinyStatistician.var(x)
        if v is None:
            return None
        return (v**.5)
