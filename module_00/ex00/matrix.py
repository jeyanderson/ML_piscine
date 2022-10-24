from urllib.parse import _NetlocResultMixinBytes


class Matrix:
    @staticmethod
    def type_is_consistent(iterable):
        types=list(set(map(type,iterable)))
        if len(types)==1 and types[0]==list:
            return list
        elif len(types)>1 and list in types:
            return None
        elif len(types)> ((int in types)+(float in types)):
            return None
        else:
            return types[0]
    @staticmethod
    def is_valid(arg):
        queue=arg.copy()
        n_dim=0
        while queue:
            n_dim+=1
            consistent=Matrix.type_is_consistent(queue)
            if consistent is None:
                print('list can only contain elements of the same type.')
                return False
            n=len(queue)
            if consistent is list:
                if len(set(map(len,queue)))>1:
                    print('shape is inconsistent.')
                    return False
                for _ in range(n):
                    elem=queue.pop(0)
                    queue.extend(elem)
            else: 
                break
        return n_dim
    def __init__(self,init):
        self.data=None
        self.shape=None
        if isinstance(init,list):
            res=self.is_valid(init)
            if not res:
                return
            if res!=2:
                print('Matrix can only be initialized with a list of list or a tuple.')
                return
            self.data=init
            self.shape=(len(init),len(init[0]))
        elif isinstance(init,tuple):
            err_msg='Matrix from shape setup can only be used providing 2 positive integers: (n_rows, n_columns).'
            if len(init)!=2:
                print(err_msg)
                return
            if not isinstance(init[0],int) or not isinstance(init[1],int):
                print(err_msg)
                return
            (n,m)=init
            if n<=0 or m<=0:
                print(err_msg)
                return
            self.data=[[0 for j in range(m)] for i in range(n)]
            self.shape=init
        else:
            print('Matrix can only be initialized with a list of list or a tuple.')
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        if self.data:
            return f'{type(self).__name__}({self.data})'
        return 'None'
    def __add__(self,op):
        if not self.data:
            return None
        err_msg='only Matrix/Vector of the same size can be added together.'
        if isinstance(op,type(self)):
            if not op.data:
                return None
            if self.shape != op.shape:
                print(err_msg)
                return None
            (n,m)=self.shape
            data=[[0 for j in range(m)]for i in range(n)]
            for j in range(m):
                for i in range(n):
                    data[i][j]+=op.data[i][j]+self.data[i][j]
            return type(self)(data)
        else:
            print(err_msg)
    def __radd__(self,op):
        return self.__add___(op)
    def __sub__(self,op):
        if not self.data:
            return None
        err_msg='only Matrix/Vector of the same size can be substracted together.'
        if isinstance(op,type(self)):
            if not op.data:
                return None
            if self.shape != op.shape:
                print(err_msg)
                return None
            (n,m)=self.shape
            data=[[0 for j in range(m)]for i in range(n)]
            for j in range(m):
                for i in range(n):
                    data[i][j]+=op.data[i][j]-self.data[i][j]
            return type(self)(data)
        else:
            print(err_msg)
    def __rsub__(self,op):
        return self.__sub__(op)
    def __truediv__(self,scalar):
        if not self.data:
            return None
        err_msg='Matrix/Vector can only be divided by scalars.'
        if not isinstance(scalar,(int,float)):
            print(err_msg)
            return None
        if not scalar:
            print('Division by zero not allowed.')
            return None
        (n,m)=self.shape
        data=[[self.data[i][j]/scalar for j in range(m)]for i in range(n)]
        return type(self)(data)
    def __rtruediv__(self,scalar):
        if not self.data:
            return None
        err_msg='Matrix/Vector can only be divided by scalars.'
        if not isinstance(scalar,(int,float)):
            print(err_msg)
            return None
        if not scalar:
            print('Division by zero not allowed.')
            return None
        (n,m)=self.shape
        data=[[0 for j in range(m)]for i in range(n)]
        for i in range(n):
            for j in range(m):
                if not self.data[i][j]:
                    print('Divison by zero not allowed.')
                    return None
                data[i][j]+=scalar/self.data[i][j]
        return type(self)(data)
    def __mul__(self,op):
        if not self.data:
            return None
        err_msg='Matrix can only be multiplied by another Matrix, a Vector or a scalar.'
        if isinstance(op,(int,float)):
            (n,m)=self.shape
            data=[[self.data[i][j]*op for j in range(m)]for i in range(n)]
            return type(self)(data)
        elif type(self)is Matrix and type(op)is Vector:
            if not op.data:
                return None
            (n,m)=self.shape
            d=op.shape[0]
            if m!=d:
                print('number of columns of the Matrix doesnt match the number of rows of the Vector.')
                return None
            data=[[sum([self.data[i][j]*op.data[j][0]for j in range(m)])]for i in range(n)]
            return Vector(data)
        elif isinstance(op,Matrix):
            if not op.data:
                return None
            (n,m)=self.shape
            (p,q)=op.shape
            if m!=p:
                print('the number of columns of the left operand dowsnt match the number of rows of the right one.')
                return None
            data=[[sum([self.data[i][k]*op.data[k][j] for k in range(m)])for j in range(q)]for i in range(n)]
            return Matrix(data)
        else:
            print(err_msg)
            return None
    def __rmul__(self,op):
        if not self.data:
            return None
        err_msg='Matrix can only be multiplied by another Matrix, a Vector or a scalar.'
        if isinstance(op,(float,int)):
            return self.__mul__(op)
        elif isinstance(op,Matrix):
            return op.__mul__(self)
        else:
            print(err_msg)
            return None
    def T(self):
        if not self.data:
            return None
        (n,m)=self.shape
        data=[[self.data[i][j] for i in range(n)]for j in range(m)]
        return type(self)(data)
class Vector(Matrix):
    def __init__(self,init):
        super().__init__(init)
        if self.shape:
            (n,m)=self.shape
            if n!=1 and m!=1:
                print('Vector can only be initialized with a list of shape (1,n) or (n,1).')
                return
    def dot(self,vec):
        try:
            assert isinstance(vec,Vector),'You can calculate a dot product only between two Vectors of the same shape.'
            assert self.shape==vec.shape,'You can calculate a dot product only between two Vectors of the same shape.'
        except Exception as e:
            print(e)
        else:
            if (self.shape[0]==1):
                return sum([x*y for(x,y)in zip(self.data[0],vec.data[0])])
            elif (self.shape[1]==1):
                return self.T().dot(vec.T())

        