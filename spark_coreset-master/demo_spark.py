#!/usr/bin/env python
__author__ = 'Ahmad Yasin & Anton Boyko'
import numpy as np, pandas as pd, time, csv, string
from cStringIO import StringIO
import tree, do_kmeans, utils, graph #,spark_kmeans_python
from coreset import Coreset
from weighted_kmeans import KMeans
import pyspark.mllib.clustering

# common parameters
inc=10; epsln=0.0001; iStepRef=10; K=1000; fxpartitions = xx = 4 #[2,4,8,16,32,64]
rng    = [1, 0,40,inc]; rngstr="%d..%d:%d"%(rng[1],rng[2],rng[3]); #rngstr = 'custom!' ###
k      = 14  ### 2,20,1
trials = 5  ## 1,50,3
rnds   = rngstr ###
iStep  = 5; iStepCS=1  ## 5: 1,20,1
#xx     = rngstr ### 2,40,2
# coreset parameters
csize       = 5*K #30k ### 10*K,300*K,10*K ]; rngstr="(%d..%d:%d)xK"%(rng[1]/K,rng[2]/K,rng[3]/K)
csg_isteps  = 1 ### 3: 1,20,1
csg_rounds  = 3 ### 0,100,5
zloop       = 7 ### 1,50,5 trials=1
run = {'spk': 1, 'rnd': 0} ###
run['uni'] = 2 #uniform
plot= {'dataset': 0, 'result': 0}
inmem  = 8 #TODO: maintain inmem=0
xlabel0 = "#iterations "
xlabel  = xlabel0 + ("   /  fxSpk%d inM%d  "%(not run['spk'], inmem)); testk = isinstance(k, basestring); testx = (xx is not fxpartitions)

if __name__ == "__main__":
    def getData(iterator):
        return pd.read_csv(StringIO("\n".join(iterator)), header=None, delim_whitespace=True, dtype=np.float64).as_matrix()
        
    def parsePartition_for_cost(iterator):
        return [getData(iterator)]

    def mapForCost(arr):
        return np.sum(utils.get_dist_to_centers(arr, means1))

    def readPointBatch(Int,iterator):
        return [(Int/2, [getData(iterator), None])]

    def merge(a ,b):
        points = np.vstack((a[0], b[0]))
        weights = None
        if a[1] is None and b[1] is not None:            a[1] = np.ones(a[0].shape[0])
        if a[1] is not None and b[1] is None:            b[1] = np.ones(b[0].shape[0])
        if a[1] is not None and b[1] is not None:        weights = np.hstack((a[1], b[1]))
        size = (a[0].shape[0] + b[0].shape[0])/2
        return points, weights, size

    def reduce(data, size):
        c = Coreset(data[0], k, data[1])
        p, w = c.compute(csize, csg_rounds, csg_isteps)
        return [p, w] # needs to be a python iterator for spark.

    def coreset(a, b):
        p, w, size = merge(a, b)
        c = reduce([p, w], size)
        return c

    def computeTree(rdd, f):
        while rdd.getNumPartitions() != 1:
            rdd = (rdd
                   .reduceByKey(f)  # merge couple and reduce by half
                   .map(lambda x: (x[0] / 2, x[1]))  # set new keys
                   .partitionBy(rdd.getNumPartitions() / 2))    # reduce num of partitions
        return rdd.reduceByKey(f).first()[1] #if not a complete tree, first is actually everything now.. #return the corest as a numpy array

    def r(x, d=2):
        return round(x, d)
        
    def s(x):
        return str(x).replace("\n", ";").replace("   ", " ").replace(
            "array(", "").replace("),", ";").replace(",", " ").replace("[ ","[").replace("  ", " ")
        
    
    def load_coreset():
        return sc.textFile(infile, xx).mapPartitionsWithIndex(readPointBatch) # from text file to (key,numpy_array)
        
    def ref_kmeans(k):
        cost, ref_means, n = do_kmeans.do_kmeans(k, infile, n_init=iStepRef, plot=plot['dataset'])    
        data.append([1, k if testk else "reference", n, cost, "n/a", s(ref_means), "n/a"])
        return cost, n

    # start from here.
    sc, config, infile = tree.init_spark()
    data=[]; data1=[]; data2=[]; x_ax=[]; y_cs=[]; y_sp1=[]; y_sp2=[]; t_cs=[]; t_sp=[]; t_csg=[]; runtg=[]
    y_un=[]; t_un=[]; #uniform
    ref_cost = 0
    
    xlabel += infile
    runtg.append(xlabel0)
    runtg.append('k='+str(k) + '; iStep='+str(iStep) + '; trials='+str(trials) + '; rounds='+str(rnds) + '; parts='+str(xx) + ';')
    runtg.append('coreset gen: size='+str(csize) + ';iStep='+str(csg_isteps) + ';rounds='+str(csg_rounds) + ';zloop='+str(zloop) )
    runtag = string.join(runtg, '@'); print '\n\n' + runtag
    total_time = time.time()
    if not testk: ref_cost, N = ref_kmeans(k)
    
    points_rdd00 = sc.textFile(infile, fxpartitions)
    points_rdd0 = points_rdd00.mapPartitions(parsePartition_for_cost).persist()
    points_rdd1 = None; coreset_rdd = None;
    if not testx:
        points_rdd1 = sc.textFile(infile, xx).map(lambda line: np.array([float(x) for x in line.split(' ')])).persist()
        coreset_rdd = load_coreset().persist()
        
    def run_spark(rnds, iStep, initMode="k-means||"):
        t = time.time()
        if inmem:
            skp_clusters = spark_kmeans_python.skp_rdd(k, points_rdd1, xx, rnds, initMode, initSteps=iStep)
        else:
            skp_clusters,tmp = spark_kmeans_python.skp(k, sc, infile, xx, rnds, initMode, initSteps=iStep)
        time1 = time.time() - t
        means1 = np.array(clusters1.centers, dtype=np.float64)
        p = points_rdd0.map(mapForCost) #.mapPartitions(parsePartition_for_cost)
        cost = p.reduce(lambda a, b : a+b)
        return cost, time1, means1

    def custom_rng(n=1310000):
        r = [];
        r.append(int(n * 0.0001))
        for i in range(1, 10):
            r.append(int(n * i* 0.001))
        for i in range(1, 11):
            r.append(int(n * i* 0.01))
        return r

    a_range = custom_rng(N) if rngstr is 'custom!' else range(rng[1], rng[2]+1, rng[3])
    a_range.insert(0, a_range[0]); loaded=False
    for var in a_range:
        rnds = var #k #csize #xx #csg_rounds #csg_isteps #iStep #k #zloop #trials ###
        fxspk = (var == rng[1]) or (var == (rng[1]+inc))
        cor_avg = uni_avg = weight_avg = cs_mis = sp1_mis = tavg_cs = tavg_csg = 0
        if run['spk'] or fxspk:  sp2_mis = tavg_sp = 0
        if run['uni']>1 or (run['uni']==1 and fxspk): uni_mis = tavg_un = 0
        if testk:
            ref_cost, tmp = ref_kmeans(k)
        if testx:
            coreset_rdd = load_coreset().persist()
            points_rdd1 = sc.textFile(infile, xx).map(lambda line: np.array([float(x) for x in line.split(' ')])).persist()
        for i in range(0,trials):            
            t = time.time() 
            result = (computeTree(coreset_rdd if inmem else load_coreset(), coreset))
            time_coreset = time.time() - t
            points = result[0]
            weights = result[1]
            weight_avg += np.sum(weights)
            
            ##coreset
            cs_cost = float("inf")
            cs_means = []
            time_kmeans = time_cost = 0
            for z in range(0,zloop):
                t = time.time()
                means1 = KMeans(points, np.expand_dims(weights, axis=0), k, rounds=rnds, n_init=iStepCS).compute()
                time_kmeans += (time.time() - t)
                t = time.time()
                #p = points_rdd.mapPartitions(parsePartition_for_cost)
                p = points_rdd0.map(mapForCost)
                a_cost = p.reduce(lambda a, b : a+b)
                time_cost += (time.time() - t)
                if (a_cost<cs_cost):
                    cs_cost = a_cost
                    cs_means = means1
            cor_avg += cs_cost
            mis1 = (1 - ref_cost / cs_cost); cs_mis += mis1
            
            size = points.shape[0] # random sampling per size of the coreset
            ##uniform
            if run['uni']>1 or (run['uni']==1 and fxspk):
                p = points_rdd00.takeSample(False, size)
                a = getData(p)
                t = time.time()
                means1 = KMeans(a, np.expand_dims(np.ones(size), axis=0), k, rounds=rnds).compute(False)
                tavg_un += (time.time() - t)
                p = points_rdd0.map(mapForCost)
                uni_cost = p.reduce(lambda a, b : a+b)
                uni_avg += uni_cost
                uni_mis += (1 - ref_cost / uni_cost)
                
            ##spark, random
            if run['rnd'] and (run['spk'] or fxspk):
                t = time.time()
                clusters1 = pyspark.mllib.clustering.KMeans.train(points_rdd1, k, initializationMode="random", maxIterations=rnds, runs=1, initializationSteps=iStep, epsilon=epsln)
                time_spark = time.time() - t
                means1 = sp1_means = np.array(clusters1.centers, dtype=np.float64)
                p = points_rdd0.map(mapForCost)
                sp1_cost = p.reduce(lambda a, b : a+b)                
                sp1_mis += (1 - ref_cost / sp1_cost)
            
            ##spark, k-means||
            if run['spk'] or fxspk:
                #sp2_cost, time_spark, sp2_means = runi_spark(rnds, iStep)
                t = time.time()
                clusters1 = pyspark.mllib.clustering.KMeans.train(points_rdd1, k, maxIterations=rnds, runs=1, initializationSteps=iStep, epsilon=epsln)
                time_spark = time.time() - t
                means1 = sp2_means = np.array(clusters1.centers, dtype=np.float64)
                p = points_rdd0.map(mapForCost)
                sp2_cost = p.reduce(lambda a, b : a+b)                
                sp2_mis += (1 - ref_cost / sp2_cost)
                tavg_sp += time_spark
                if ref_cost is 0: ref_cost = sp2_cost
            
            if loaded:
                tavg_cs += time_kmeans; tavg_csg += time_coreset;
                data.append([xx, var, r(np.sum(weights)), cs_cost, sp2_cost, s(cs_means), s(sp2_means),
                    r(time_coreset), r(time_kmeans), r(time_cost), r(time_spark)])
        #trials loops end
        if loaded:
            data1.append([xx, var, r((weight_avg)/trials), (cor_avg)/trials, (uni_avg)/trials])
            cs_mis /= trials; sp1_mis /= trials
            tavg_cs += tavg_csg; tavg_cs /= trials; tavg_csg /= trials
            if run['spk'] or fxspk: sp2_mis /= trials; tavg_sp /= trials
            if run['uni']>1 or (run['uni']==1 and fxspk): uni_mis /= trials; tavg_un /= trials
            data2.append([xx, var, r(weight_avg/trials), cs_mis, sp2_mis, r(tavg_cs,1), r(tavg_csg,1), r(tavg_sp,1)])
            y_cs.append(cs_mis); y_sp1.append(sp1_mis); y_sp2.append(sp2_mis)
            y_un.append(uni_mis); t_un.append(tavg_un);#uniform
            t_cs.append(tavg_cs); t_csg.append(tavg_csg); t_sp.append(tavg_sp); x_ax.append(var)
        else:
            loaded = True
            xlabel += ' ref-cost=%d'%ref_cost

    total_time = (time.time() - total_time)
    def dump(nam, dat, hdr1, hdr2):
        with open('results_' + nam + '.csv', 'w') as fp:
            a = csv.writer(fp, delimiter=',')
            a.writerow(hdr1); a.writerow(hdr2)
            a.writerows(dat)
    
    dump('all', data,  [runtag, 'ref-cost: ', ref_cost, ' total-time:', r(total_time, 1)],
        ['partitions', 'VAR', 'sum u(x)', 'cor_cost', 'sp2_cost', 'cor_means', 'sp2_means', 'time_coreset', 'time_kmeans', 'time_cost', 'time_spark'])
    dump('avg', data1, [runtag, 'ref-cost: ', ref_cost, ' total-time:', r(total_time, 1)],
        ['partitions', 'VAR', 'coreset weights avg', 'coreset avg cost', 'uniform avg cost'])
    dump('mis', data2, [runtag, 'ref-cost:', ref_cost],
        ['partitions', 'VAR', 'cset weights avg', 'cset mistake', 'spark mis', 'time cset', 'time spark'])

    un = [y_un, t_un] if run['uni'] else None;
    graph.plot(x_ax, [y_cs, t_cs, t_csg], [y_sp2, t_sp], None, un, show=plot['result'], labels=[xlabel, runtg[1], runtg[2]])

