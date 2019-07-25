'''

Student Name: Alon Amar (replace with your name)
GT User ID: aamar32 (replace with your User ID)
GT ID: 903339940 (replace with your GT ID)

'''
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode as mrk
import StrategyLearner as sl
import ManualStrategy as ms


def author():
    return 'aamar32'

def test_code():
    sym = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    learner = sl.StrategyLearner(verbose=False, impact=0.005)  # constructor
    learner.addEvidence(symbol=sym, sd=sd, ed=ed, sv=100000)  # training phase

    orders = learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=100000)  # testing phase
    slDF = mrk.tranformOrders(orders)
    slDF = mrk.compute_portvals(slDF, 100000, commission=0.0, impact=0.005)
    norm_slDF = slDF / slDF.iloc[0]

    orders = ms.testPolicy(symbol=sym, sd=sd, ed=ed, sv=100000)
    msDF = mrk.tranformOrders(orders)
    msDF = mrk.compute_portvals(msDF, 100000, commission=0.0, impact=0.005)
    norm_msDF = msDF / msDF.iloc[0]

    orders = ms.testPolicyOpt(symbol=sym, sd=sd, ed=ed, sv=100000)
    optDF = mrk.tranformOrders(orders)
    optDF = mrk.compute_portvals(optDF, 100000, commission=0.0, impact=0.005)
    norm_optDF = optDF / optDF.iloc[0]

    bench_DF = mrk.benchmark("JPM", sd=sd, ed=ed, sv=100000)
    norm_bench_DF = bench_DF / bench_DF.iloc[0]


    plt.figure(figsize=(14, 6))
    plt.title("Experiment 1: Manual vs. Q-Learner")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.plot(norm_optDF, 'green', label="Optimal")
    plt.plot(norm_slDF, 'red', label="Q-Learner")
    plt.plot(norm_msDF, 'black', label="Manual")
    plt.plot(norm_bench_DF, 'blue', label="Benchmark")
    plt.tick_params(axis='x', labelrotation=25)
    plt.legend()
    plt.grid()
    plt.savefig("Experiment1.png", dpi=300)

    opt_cr, opt_adr, opt_sddr, opt_sr = mrk.calc_stats(optDF['SUM'])
    q_cr, q_adr, q_sddr, q_sr = mrk.calc_stats(slDF['SUM'])
    man_cr, man_adr, man_sddr, man_sr = mrk.calc_stats(msDF['SUM'])
    bench_cr, bench_adr, bench_sddr, bench_sr = mrk.calc_stats(bench_DF['SUM'])
    # Compare portfolio against $SPX
    # print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Optimal: {}".format(opt_sr)
    print "Sharpe Ratio of Learner: {}".format(q_sr)
    print "Sharpe Ratio of Manual: {}".format(man_sr)
    print "Sharpe Ratio of Benchmark: {}".format(bench_sr)
    print
    print "Cumulative Return of Optimal: {}".format(opt_cr)
    print "Cumulative Return of Learner: {}".format(q_cr)
    print "Cumulative Return of Manual: {}".format(man_cr)
    print "Cumulative Return of Benchmark: {}".format(bench_cr)
    print
    print "Standard Deviation of Optimal: {}".format(opt_sddr)
    print "Standard Deviation of Learner: {}".format(q_sddr)
    print "Standard Deviation of Manual: {}".format(man_sddr)
    print "Standard Deviation of Benchmark: {}".format(bench_sddr)
    print
    print "Average Daily Return of Optimal: {}".format(opt_adr)
    print "Average Daily Return of Learner: {}".format(q_adr)
    print "Average Daily Return of Manual: {}".format(man_adr)
    print "Average Daily Return of Benchmark: {}".format(bench_adr)


if __name__ == "__main__":
    test_code()