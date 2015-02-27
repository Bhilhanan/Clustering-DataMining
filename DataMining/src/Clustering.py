
import time
from Orange import distance, clustering
from Orange import orange
import Orange.data
import Orange.clustering 
import os
import numpy
from methods import *



f1='Hierarchical.csv'
if os.path.exists(f1):
    os.remove(f1)
    
f2='KMeans.csv'
if os.path.exists(f2):
    os.remove(f2)
    
data=Orange.data.Table('output.tab')
#matrix = Orange.misc.SymMatrix(len(data))
numDocs=len(data)
print "Count of documents in Reuters dataset: " +str(numDocs) + "\n"
print "1. Constructing Distance Matrices\n"

starter=time.time()
constructorEuclidean=distance.Euclidean()
EuclideanDistanceMat=distance.distance_matrix(data, distance_constructor=constructorEuclidean)
euclidean_hierarchical_clustering=clustering.hierarchical.HierarchicalClustering()
euclidean_hierarchical_clustering.linkage=clustering.hierarchical.AVERAGE
euclideanRoot=euclidean_hierarchical_clustering(EuclideanDistanceMat)
ender=time.time()
timer=ender-starter


starter1=time.time()
constructorManhattan=distance.Manhattan()
ManhattanDistanceMat=distance.distance_matrix(data, distance_constructor=constructorManhattan)
manhattan_hierarchical_clustering = clustering.hierarchical.HierarchicalClustering()
manhattan_hierarchical_clustering.linkage=clustering.hierarchical.AVERAGE
manhattanRoot=manhattan_hierarchical_clustering(ManhattanDistanceMat)
ender1=time.time()
timer1=ender1-starter1

print "2. Time: Hierarchical clustering: "
print "With Euclidean distance = " +str(timer)+ "sec"
print "With Manhattan distance = " +str(timer1)+ "sec\n"

euclideanRoot.mapping.objects=data
manhattanRoot.mapping.objects=data

topmost=128    
file1 = open('Hierarchical.csv', 'a')
file1.write("Time (clustering using Euclidean distance): "+str(timer)+"\n"+"Time (clustering using Manhattan Distance): "+str(timer1)+"\n"+"Total Clusters :" +str(topmost)+"\n\n")
file1.write("#Clusters,")
file1.write("EntropyEuclidean,,")
file1.write("SkewEuclidean(std dev),,,")
file1.write("EntropyManhattan,,")
file1.write("SkewManhattan(std dev),")
file1.write("\n")
clusterListEuclidean=[]
clusterListEuclidean.append(euclideanRoot.left)
clusterListEuclidean.append(euclideanRoot.right)
heightEuclidean=euclideanRoot.height
clusterListManhattan=[]
clusterListManhattan.append(manhattanRoot.left)
clusterListManhattan.append(manhattanRoot.right)
heightManhattan=manhattanRoot.height



print "3. Hierarchical Clustering: Calculating Entropy and Skew\n"

for x in range(topmost):
    #######################Hierarchical Euclidean##################################################3
    file1.write(str(len(clusterListEuclidean)-1)+",")
    indexCluster=0
    clustersAssignEuclidean=[0]*(numDocs+1)
    for cluster in clusterListEuclidean:
        for point in cluster:
            clustersAssignEuclidean[int(str(point["Documents"]))]=indexCluster
        indexCluster+=1
    dist=getDistMultiClass(data, clustersAssignEuclidean, len(clusterListEuclidean))
    classesNum=len(dist)

    entropy=evalEntropy(dist, clustersAssignEuclidean, classesNum, numDocs)

    file1.write(str(entropy)+",,")
    stdevEuclidean=evalStdDeviation(clustersAssignEuclidean, classesNum)

    file1.write(str(stdevEuclidean)+",,,")
    heightEuclidean=-1
    for cluster in clusterListEuclidean:
        if cluster.height>heightEuclidean:
            splitClusters= [cluster]
            heightEuclidean=cluster.height
        elif cluster.height==heightEuclidean:
            splitClusters.append(cluster)
            
    for splitCluster in splitClusters:
        clusterListEuclidean.remove(splitCluster)
        clusterListEuclidean.append(splitCluster.left)
        clusterListEuclidean.append(splitCluster.right)


#######################Hierarchical Manhattan##################################################

    indexCluster=0
    clustersAssignManhattan=[0]*(numDocs+1)
    for cluster in clusterListManhattan:
        for point in cluster:
            clustersAssignManhattan[int(str(point["Documents"]))] =indexCluster
        indexCluster+=1
    dist=getDistMultiClass(data, clustersAssignManhattan, len(clusterListManhattan))
    classesNum=len(dist)
    entropy=evalEntropy(dist, clustersAssignManhattan, classesNum, numDocs)
    file1.write(str(entropy)+",,")
    stdevManhattan=evalStdDeviation(clustersAssignManhattan, classesNum)
    file1.write(str(stdevManhattan)+",")
    file1.write("\n")
    heightManhattan=-1
    for cluster in clusterListManhattan:
        if cluster.height>heightManhattan:
            splitClusters= [cluster]
            heightManhattan=cluster.height
        elif cluster.height==heightManhattan:
            splitClusters.append(cluster)
            
    for splitCluster in splitClusters:
        clusterListManhattan.remove(splitCluster)
        clusterListManhattan.append(splitCluster.left)
        clusterListManhattan.append(splitCluster.right)
        
file1.close()  

clustersNum=128
starter2=time.time()
euclidean_kmeans=clustering.kmeans.Clustering(data,clustersNum, distance=distance.Euclidean)
ender2=time.time()
timer2=ender2-starter2

starter3=time.time()
manhattan_kmeans=clustering.kmeans.Clustering(data,clustersNum, distance=distance.Manhattan)
ender3=time.time()
timer3=ender3-starter3

print "4. Time: KMeans clustering: "
print "With Euclidean distance = " +str(timer2)+ "sec"
print "With Manhattan distance = " +str(timer3)+ "sec\n"

print "5. KMeans Clustering: Calculating Entropy and Skew\n"

#########################################KMeans Clustering######################################################33
file2 = open('KMeans.csv', 'a')
file2.write("Time (clustering using Euclidean Distance): "+str(timer2)+"\n"+"Time (clustering using Manhattan Distance): "+str(timer3)+ "\n"+"Total Clusters: "+str(clustersNum)+"\n\n")
file2.write("K,")
file2.write("EntropyEuclidean,,")
file2.write("SkewEuclidean(std dev),,,")
file2.write("EntropyManhattan,,")
file2.write("SkewManhattan(std dev),")
file2.write("\n")
clustersAssignEuclidean=euclidean_kmeans.clusters
clustersAssignManhattan=manhattan_kmeans.clusters


for i in range(7):
    #####################KMeans Euclidean###################################
    file2.write(str(clustersNum/(2**i))+",")
    dist=getDistMultiClass(data, clustersAssignEuclidean, clustersNum)
    classesNum=len(dist)

    entropy=evalEntropy(dist, clustersAssignEuclidean, classesNum, numDocs)

    file2.write(str(entropy)+",,")
    stdevEuclidean=evalStdDeviation(clustersAssignEuclidean, classesNum)

    file2.write(str(stdevEuclidean)+",,,")
    euclidean_kmeans=clustering.kmeans.Clustering(data,(clustersNum/(2**i)), distance=distance.Euclidean)
    clustersAssignEuclidean=euclidean_kmeans.clusters

    ####################KMEans Manhattan#############################################333333
# 
    dist=getDistMultiClass(data, clustersAssignManhattan, clustersNum)
    classesNum=len(dist)

    entropy=evalEntropy(dist, clustersAssignManhattan, classesNum, numDocs)
#    ,
    file2.write(str(entropy)+",,")
    stdevManhattan=evalStdDeviation(clustersAssignManhattan, classesNum)
#
    file2.write(str(stdevManhattan)+",")
    file2.write("\n")
    manhattan_kmeans=clustering.kmeans.Clustering(data,(clustersNum/(2**i)), distance=distance.Manhattan)
    clustersAssignManhattan=manhattan_kmeans.clusters
file2.close()
print "################################Finished#########################################"
  
    
            
      
    
