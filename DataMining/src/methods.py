
import math
import numpy


def getDistMultiClass(data, clustersAssign, clustersNum):
    dist=[]
    for x in range(clustersNum):
        var={}
        dist.append(var)
    var={}
    for x in range (len(clustersAssign)-1):
        getData=data[x].get_class().value
        getClasses=getData.split(" ")
        for y in getClasses:
            if dist[clustersAssign[x]].has_key(y):
                dist[clustersAssign[x]][y]+=1
            else:
                dist[clustersAssign[x]][y]=1
    return dist

def evalStdDeviation(clustersAssign, classesNum):
    totCount = dict(zip(clustersAssign, map(clustersAssign.count, clustersAssign))).values()
#     mean = sum(totCount) *1.0/ len(totCount)
#     return (mean-numpy.median(totCount))/numpy.std(totCount);
    return numpy.std(totCount)

def evalEntropy(dist, clustersAssign, classesNum, numDocs):
    entr=0
    for i in range(len(dist)):
        x=dist[i]
        
        classDistribution=x.values()
        base=len(classDistribution)
        if base==1:
            continue
        tot=clustersAssign.count(i)
        if tot==0:
            continue
        e=0
        for a in classDistribution:
            p=(a*1.0)/tot
            if p !=0:
                e-=1*p*1.0*math.log(p,2)
        entr+=e*tot*1.0/numDocs
    return entr