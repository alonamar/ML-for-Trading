'''

Student Name: Alon Amar (replace with your name)
GT User ID: aamar32 (replace with your User ID)
GT ID: 903339940 (replace with your GT ID)

'''
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode as mrk
import StrategyLearner as sl
import numpy as np
import ManualStrategy as ms

def author():
    return 'aamar32'

def test_code():
    sym = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    impacts = np.arange(0, 0.5, 0.05)
    num_orders = []
    q_cr_arr = []
    q_sr_arr = []
    for impact in impacts:
        print impact
        learner = sl.StrategyLearner(verbose=False, impact=impact)  # constructor
        learner.addEvidence(symbol=sym, sd=sd, ed=ed, sv=100000)  # training phase
        orders = learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=100000)  # testing phase
        num_orders.append(len(orders[orders[sym]!=0]))
        slDF = mrk.tranformOrders(orders)
        slDF = mrk.compute_portvals(slDF, 100000, commission=0.0, impact=0.005)
        q_cr, _, _, q_sr = mrk.calc_stats(slDF['SUM'])
        q_cr_arr.append(q_cr)
        q_sr_arr.append(q_sr)

    plt.figure(figsize=(14, 6))
    plt.title("Experiment 2: Impact Influence")
    plt.xlabel("Impact")
    plt.ylabel("Normalized Value")
    plt.plot(impacts, np.true_divide(num_orders,num_orders[0]), label="num_orders")
    plt.plot(impacts, np.true_divide(q_cr_arr,q_cr_arr[0]), label="cr")
    plt.plot(impacts, np.true_divide(q_sr_arr,q_sr_arr[0]), label="sr")
    plt.legend()
    plt.grid()
    plt.savefig("Experiment2.png", dpi=300)

    print "impact, #trades, sr, cr"
    for i in range(len(impacts)):
        print "{}, {}, {}, {}".format(impacts[i], num_orders[i], q_sr_arr[i], q_cr_arr[i])


if __name__ == "__main__":
    test_code()
    
