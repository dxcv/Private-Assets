import matplotlib
matplotlib.use('qt5agg') #qt5aggg or macosx
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from time import time
from Correlated_Brownian_Motion import correlated_brownian_motion



class Contributions():

    def __init__(self,mean_drawdown_rate,N,sigma_drawdown,Io,dt,sims,cum_drawdown,dWmkt,dWdd):
        self.mean_drawdown_rate = mean_drawdown_rate
        self.N = N
        self.sigma_drawdown = sigma_drawdown
        self.Io = Io
        self.dt = dt
        self.sims = sims
        self.dWmkt = dWmkt
        self.dWdd = dWdd
        self.cum_drawdown = cum_drawdown


    def calculate_drawdowns(self):
        drawdown_val_range = []
        drawdown_cum_range = []

        for i in range(1, int(N + 1)):
            drawdown_rate = max((self.mean_drawdown_rate * dt + self.sigma_drawdown * self.dWmkt[i - 1]),0)
            drawdown_val = drawdown_rate * (self.Io - self.cum_drawdown)
            self.cum_drawdown += round(drawdown_val,1)
            drawdown_val_range.append(drawdown_val)
            drawdown_cum_range.append(self.cum_drawdown)

        return drawdown_cum_range, drawdown_val_range



class Distributions():

    def __init__(self,mean_rate_of_return_fund,mean_rate_of_return_market,fund_value,cum_distribution,market_beta,sigma_market,sigma_idosyncratic,sigma_distribution,mean_distribution_rate,N,
                 sims,ncf,cncf,dWmkt,dWdist,dWidio):
        self.mean_rate_of_return_fund = mean_rate_of_return_fund
        self.mean_rate_of_return_market = mean_rate_of_return_market
        self.fund_value = fund_value
        self.cum_distribution = cum_distribution
        self.market_beta = market_beta
        self.sigma_market = sigma_market
        self.sigma_idosyncratic = sigma_idosyncratic
        self.sigma_distribution = sigma_distribution
        self.mean_distribution_rate = mean_distribution_rate
        self.N = N
        self.sims = sims
        self.dWmkt = dWmkt
        self.dWidio = dWidio
        self.dWdist = dWdist
        self.ncf = ncf
        self.cncf = cncf


    def calculate_distribution_and_fundvalues(self,drawdown_val_range):

        dVt = []; fund_brownian_drift_range = []; Vt = []; distribute_rate = []
        dDistribute = []; cum_distribution_range = []; ncf_range = []
        cncf_range = []; market_range = []
        market = 100
        market_range.append(market)

        for i in range(1, int(N + 1)):
            Fund_Brownian_Drift = (self.mean_rate_of_return_fund * dt + self.market_beta * self.sigma_market * self.dWmkt[i - 1] + self.sigma_idosyncratic * self.dWidio[i - 1])
            fund_brownian_drift_range.append(Fund_Brownian_Drift)

            distribution_rate = self.mean_distribution_rate * dt + self.sigma_distribution * self.dWdist[i - 1]
            distribution_rate = max(distribution_rate,0)
            distribute_rate.append(distribution_rate)

            distribute_value = self.fund_value * distribution_rate
            self.cum_distribution += distribute_value
            dDistribute.append(distribute_value)
            cum_distribution_range.append(self.cum_distribution)

            delta_fundvalue = self.fund_value * Fund_Brownian_Drift + drawdown_val_range[i - 1] - distribute_value
            self.fund_value += delta_fundvalue
            dVt.append(delta_fundvalue)
            Vt.append(self.fund_value)

            self.ncf = (-1) * (drawdown_val_range[i - 1] - distribute_value)
            ncf_range.append(self.ncf)
            self.cncf += self.ncf
            cncf_range.append(self.cncf)

            #market movement
            market += market * (self.mean_rate_of_return_market * dt + self.sigma_market * self.dWmkt[i - 1])
            market_range.append(market)


        return Vt, dDistribute, ncf_range, cncf_range, cum_distribution_range,market_range



if __name__ == '__main__':

    #Contribution
    sims = 1000
    mean_drawdown_rate = 0.41; N = 20 * 4; sigma_drawdown = .21; Io = 100.; dt = 0.25
    cum_drawdown = 0

    #Distribution
    mean_rate_of_return_fund = 0.08; fund_value = 0; cum_distribution = 0; market_beta = 1.3; sigma_market = .25
    sigma_idosyncratic = .35; sigma_distribution = .11; mean_distribution_rate = 0.08
    ncf = 0; cncf = 0; mean_rate_of_return_market = 0.11

    '''
    cc_dd_df       ###Cummulative capital drawdown rate
    qcd_df         ###Qtrly capital drawdown rate
    cc_dis_df      ###Cummulative capital distribution rate
    qc_dis_df      ###Qtrly capital distribution rate
    cncf_df        ###Cummulative net cashflow
    qncf_df        ###Qtrly net cashflow 
    '''


    cc_dd_df = pd.DataFrame(); qcd_df = pd.DataFrame(); cc_dis_df = pd.DataFrame()
    qc_dis_df = pd.DataFrame(); cncf_df = pd.DataFrame(); qncf_df = pd.DataFrame()
    Vt_df = pd.DataFrame(); market_df = pd.DataFrame()

    time_to_target_range = []; depth_jcurve_range = []
    time_to_breakeven_range = []; return_in_5yr_range = []

    for i in range(1, sims + 1):

        # Brownian motion with correlation to the market movement
        dWmkt, dWdd, dWdist, dWidio = correlated_brownian_motion(rho_mktdd=.5, rho_mkt_dist=.8, rho_mkt_idio=0, dt=0.01, n=N)

        '''
        ###############
        drawdowns
        ###############
        '''
        pa_contribute = Contributions(mean_drawdown_rate,N,sigma_drawdown,Io,dt,sims,cum_drawdown,dWmkt,dWdd)
        drawdown_cum_range,drawdown_val_range = pa_contribute.calculate_drawdowns()


        '''
        ###############
        distributions
        ###############
        '''
        pa_distribute = Distributions(mean_rate_of_return_fund,mean_rate_of_return_market, fund_value, cum_distribution, market_beta,sigma_market,
                                          sigma_idosyncratic, sigma_distribution, mean_distribution_rate, N, sims, ncf,
                                          cncf,dWmkt,dWdist,dWidio)
        Vt, dDistribute, ncf_range, cncf_range, cum_distribution_range,market_range = pa_distribute.calculate_distribution_and_fundvalues(
            drawdown_val_range)



        drawdown_cum_range_df = pd.DataFrame(drawdown_cum_range)
        cc_dd_df = pd.concat([cc_dd_df, drawdown_cum_range_df], axis=1)

        drawdown_val_range_df = pd.DataFrame(drawdown_val_range)
        qcd_df = pd.concat([qcd_df, drawdown_val_range_df], axis=1)

        cum_distribution_range_df = pd.DataFrame(cum_distribution_range)
        cc_dis_df = pd.concat([cc_dis_df, cum_distribution_range_df], axis=1)

        dDistribute_df = pd.DataFrame(dDistribute)
        qc_dis_df = pd.concat([qc_dis_df, dDistribute_df], axis=1)

        cncf_range_df = pd.DataFrame(cncf_range)
        cncf_df = pd.concat([cncf_df, cncf_range_df], axis=1)

        ncf_range_df = pd.DataFrame(ncf_range)
        qncf_df = pd.concat([qncf_df, ncf_range_df], axis=1)

        market_range.pop()
        market_range_df = pd.DataFrame(market_range)
        market_df = pd.concat([market_df,market_range_df],axis=1)



        '''time to target'''
        time_to_target = drawdown_cum_range_df.round().values.tolist().index([100])
        time_to_target_range.append(time_to_target)

        '''depth of the j-curve'''
        depth_jcurve = np.min(cncf_range)
        depth_jcurve_range.append(depth_jcurve)

        '''time to breakeven'''
        try:
            time_to_breakeven = next(x[0] for x in enumerate(cncf_range) if x[1] > [0])
        except StopIteration:
            pass
        time_to_breakeven_range.append(time_to_breakeven)

        '''return in 5 years'''
        return_in_5yr_range.append(cncf_range[20])

        '''Fund value'''
        Vt_range_df = pd.DataFrame(Vt)
        Vt_df = pd.concat([Vt_df, Vt_range_df], axis=1)



    '''
    #########
    Print all graphs
    #########
    '''

    fsize = 9
    #time to target
    plt.subplot(2, 2,1)
    plt.hist(time_to_target_range)
    plt.xlabel('months')
    plt.ylabel('count')
    plt.title('Time to target', fontsize=fsize)

    #depth of j-curve
    #plt.scatter(np.linspace(1, len(depth_jcurve_range), len(depth_jcurve_range)), depth_jcurve_range)
    plt.subplot(2, 2, 2)
    plt.plot(depth_jcurve_range)
    plt.xlabel('months')
    plt.ylabel('capital')
    plt.title('J-curve depth', fontsize=fsize)

    #time to breakeven
    plt.subplot(2, 2, 3)
    plt.hist(time_to_breakeven_range)
    plt.xlabel('months')
    plt.ylabel('count')
    plt.title('Time to breakeven', fontsize=fsize)

    #return in 5 years
    plt.subplot(2, 2, 4)
    plt.plot(return_in_5yr_range)
    plt.xlabel('months')
    plt.ylabel('cummulative net cashflow')
    plt.title('cummulative net cashflow', fontsize=fsize)

    plt.figure()
    #plt.show()

    names_list = ['Cum capital drawdown', 'Qtrly capital drawdown', 'Cum capital distribution',
                  'Qtrly capital distribution', 'Cum Net CF', 'Qtrly NCF']
    graphs = [cc_dd_df, qcd_df, cc_dis_df, qc_dis_df, cncf_df, qncf_df]

    for x in range(1,7):
        plt.subplot(3, 2, x)
        plt.axhline(y=0, color='b', linestyle='-')
        plt.plot(graphs[x-1])
        plt.title(names_list[x-1], fontsize=fsize)

    plt.figure()
    #plt.show()

    '''Calculate % drop in Fundvalue in any given year'''
    Vdfc = Vt_df.pct_change()
    Vdfc_flatten = pd.Series(Vdfc.values.ravel('F'))

    '''Calc 5yr returns'''
    rtn_5y = pd.DataFrame()
    rtn_5y = pd.concat([cc_dis_df.iloc[20], Vdfc.iloc[20], cc_dd_df.iloc[20]], axis=1)
    rtn_5y.columns = ['Cum Dist CF', 'Fund value', 'Cum drawdown']
    rtn_5y['returns'] = (rtn_5y['Cum Dist CF'] + rtn_5y['Fund value']) / rtn_5y['Cum drawdown']
    rtn_5y.index = np.arange(0, rtn_5y.shape[0])


    #fund value
    plt.subplot(2, 2,1)
    plt.plot(Vt_df)
    plt.xlabel('Fund Value')
    plt.ylabel('Months')

    #quarterly returns hist
    plt.subplot(2, 2,2)
    Vdfc_flatten.hist(bins=2000)
    plt.xlabel('quarterly returns')
    plt.ylabel('count')

    #returns after 5 years
    plt.subplot(2, 2, 3)
    plt.plot(rtn_5y['returns'])
    plt.xlabel('count')
    plt.ylabel('Returns')

    #market movement
    plt.subplot(2, 2, 4)
    plt.plot(market_df)
    plt.xlabel('months')
    plt.ylabel('market')

    plt.figure()
    #plt.show()

    #fund and marketdrawdown
    df_returns = Vt_df.pct_change().dropna()
    cum_returns = (1 + df_returns).cumprod()
    drawdown = 1 - cum_returns.div(cum_returns.cummax())
    max_drawdown_range = np.max(drawdown)
    np.percentile(max_drawdown_range, 95)


    #market drawn hist
    plt.subplot(1, 2,1)
    max_drawdown_range.hist()
    plt.xlabel('max drawdown')
    plt.ylabel('count')


    print('Average time to target is {:.2f} months'.format(np.mean(time_to_target_range)))
    print('Average depth of J-curve is {:.2f}'.format(np.mean(depth_jcurve_range)))
    print('Average time to breakeven is {:.2f} months)'.format(np.mean(time_to_breakeven_range)))
    print('Average return after 5 years is {:.2f}%'.format(np.mean(rtn_5y['returns'])*100))
    print('95th percentile fund drawdown is {:.2f}%'.format(np.percentile(max_drawdown_range,95) * 100))


    plt.show()