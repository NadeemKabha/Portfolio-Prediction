import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import date
import itertools
import math
import yfinance as yf
from typing import List
import decimal
from decimal import *

class PortfolioBuilder:


    def get_daily_data(self, tickers_list: List[str], start_date: date,end_date: date = date.today()) -> pd.DataFrame:
        try:
            df = web.get_data_yahoo(tickers_list, start=start_date, end=end_date)['Adj Close'] # Gets the data
        except:
            raise ValueError
        if df.isnull().values.any():
            raise ValueError
        self.length=len(tickers_list) # Gives the number of tickers(for the first portfolio)
        self.b0 = [] # The first portfolio vector
        for i in range(self.length): # Gives the first portfolio its values
            self.b0.append(1 / self.length)

        self.dfarr=df.values # Transfers the data from df to array
        self.xtj=[] # This is the x vector
        for t in range(len(self.dfarr)-1):
            self.xtj.append([])
            for j in range(self.length):
                self.xtj[t].append(self.dfarr[t+1][j]/self.dfarr[t][j])

        return df
        pass


    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:

        def dot_product(x, y):
            prod = sum(x[i] * y[i] for i in range(len(x)))
            return prod
        def mvproduct(x,y):
            a=np.array(x)
            b=np.array(y)
            return (a.dot(b)).tolist()
        def scalar_product(a,x):

            prod=[]
            for i in range(len(x)):

                if (type(x[i]) ==list) or (type(x[i]) ==set)or (type(x[i]) ==tuple):
                    prod.append([])
                    for j in range (self.length):
                        prod[i].append(a*x[i][j])
                else:
                    prod.append(a * x[i])
            return prod
        def vector_sum(x):
            vsum=[]
            for j in range(self.length):
                vsum.append(sum(x[i][j] for i in range(len(x))))
            return vsum


        def frange(start, stop, step):
            while start < stop:
                getcontext().prec = 8
                yield float(start)
                start += decimal.Decimal(step)

        bw_lst = []
        fbw_lst = list(itertools.product(frange(0, 1 + 1 / portfolio_quantization, 1 / portfolio_quantization),
                                         repeat=self.length))
        for i in range(len(fbw_lst)):
            if 0.999 <= (sum(fbw_lst[i][j] for j in range(self.length))) <= 1.001:
                bw_lst.append(fbw_lst[i])




        b = []
        b.append(self.b0)  # Builds the list of portfolios with b0

        for T in range(1,len(self.dfarr)):
            b.append([])
            mone_lst=[]
            st_lst = []
            for w in range(len(bw_lst)):
                st=1

                for t in range(T):
                    st=st*(dot_product(bw_lst[w],self.xtj[t]))
                st_lst.append(st)

                mone_lst.append(scalar_product(st_lst[w],bw_lst[w]))
            mechane=1/(sum(st_lst[i] for i in range(len(st_lst))))
            b[T]=scalar_product(mechane,vector_sum(mone_lst))
        my_lst = []
        for t in range(len(self.dfarr) - 1):
            my_lst.append(dot_product(b[t], self.xtj[t]))
        wealth = [1.0]
        for T in range(1, len(self.dfarr)):
            wealth.append(np.prod(my_lst[:T]))


        return wealth

        pass


    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:

        def dot_product(x,y):
            prod=sum(x[i] * y[i] for i in range(len(x)))
            return prod

        b = []
        b.append(self.b0)  # Builds the list of portfolios with b0
        for t in range (1,len(self.dfarr)):
            b.append([])
            for j in range(self.length): # A function that calculates the portfolio in a specific day fo one product
                b[t].append((b[t-1][j]*math.exp((learn_rate*self.xtj[t-1][j])/dot_product(b[t-1],self.xtj[t-1])))/(sum(b[t-1][k]*math.exp((learn_rate*self.xtj[t-1][k])/dot_product(b[t-1],self.xtj[t-1])) for k in range (self.length))))
        my_lst=[]
        for t in range(len(self.dfarr)-1):
            my_lst.append(dot_product(b[t],self.xtj[t]))
        wealth=[1.0]
        for T in range (1,len(self.dfarr)):
            wealth.append(np.prod(my_lst[:T]))
        return wealth
        pass



