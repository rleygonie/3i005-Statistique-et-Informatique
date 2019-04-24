import math
from utils import *


def getPrior(data):
    d = data.get("target")
    dbis =  1.0 * d.sum() / d.size
    dter = dbis - 1.96 *  math.sqrt(( dbis *(1.0 - dbis)) / d.size)
    dqter =  dbis + 1.96 *  math.sqrt(( dbis *(1.0 - dbis)) / d.size)
    d = { 'estimation' : dbis, 'min5pourcent' : dter, 'max5pourcent' : dqter}
    return d
    
    
class APrioriClassifier(AbstractClassifier):
    
    def ___init__(self):
        pass
    
    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1
        
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        return 1 
    #int (getPrior(attrs).get('estimation') > 0.5)
    
    
    def statsOnDF(self, df):
        restmp = [0,0,0,0]
        for t in range(len(df)): 
            restmp[ 2 * (int) (df.iloc[t].get('target') == 1) + (int) (self.estimClass(df.iloc[t]) == 1)  ] += 1
        res = {'VP' : restmp[3] ,'VN' : restmp[0], 'FP' : restmp[1] , 'FN' : restmp[2]}
        tmp1 = 1.0*(res.get('VP')) / (res.get('VP') + (res.get('FP')))
        tmp2 = 1.0*res.get('VP') / (res.get('VP') + res.get('FN'))
        restmp2 = {'Précision' : tmp1, 'Rappel' : tmp2}
        res.update(restmp2)
        return res
    
    
    
def P2D_l(data, arg):
    databis = data.get([arg,'target'])
    listetarget1 = databis.query('target==1')
    listetarget0 = databis.query('target==0')
    listeargs = np.unique(databis.get(arg).values)
    listeGroupe = [listetarget0,listetarget1]
    lenarg = len(listeargs)
    res1 = [0]*lenarg    
    res0 = [0]*lenarg 
    resGroupe = [res0,res1]
    for target in [0,1]:
        for i in range(lenarg):
            elem = listeargs[i]   
            s = arg+"=="+str(elem)
            resGroupe[target][i]+=len(listeGroupe[target].query(s))
    dicoRes = {}
    for target in [1,0]:
        tmpDic = {}
        for i in range(lenarg):
            tmpDic2 = {listeargs[i]: 1.0* resGroupe[target][i] / len(listeGroupe[target])}
            tmpDic.update(tmpDic2)
        dicoRes[target] = tmpDic
    return dicoRes
        
    
def P2D_p(data, arg):
    listetarget = []
    listepastarget = []
    databis = data.get([arg,'target'])
    listetarget1 = databis.query('target==1')
    listetarget0 = databis.query('target==0')
    listeargs = np.unique(databis.get(arg).values)
    listeGroupe = [listetarget0,listetarget1]
    lenarg = len(listeargs)
    res1 = [0]*lenarg    
    res0 = [0]*lenarg 
    resGroupe = [res0,res1]
    for target in [0,1]:
        for i in range(lenarg):
            elem = listeargs[i]   
            s = arg+"=="+str(elem)
            resGroupe[target][i]+=len(listeGroupe[target].query(s))
    dicoRes = {}
    for i in range(lenarg):
        tmpDic = {}
        proba1 =  1.0* resGroupe[1][i] / (resGroupe[0][i] + resGroupe[1][i])
        proba0 = 1 - proba1
        tmpDic[listeargs[i]] = {1: proba1, 0: proba0}
        dicoRes.update(tmpDic)
    return dicoRes
     
class ML2DClassifier(APrioriClassifier):
    
    def __init__(self,data,arg):
        self.dico = P2D_l(data,arg)
        self.argument = arg
    
    def estimClass(self, attrs):
        #print(attrs)
        value = attrs.get(self.argument)        
        probaTarget1 = self.dico.get(1).get(value)
        probaTarget0 = self.dico.get(0).get(value)
        #print(value,probaTarget1,probaTarget0)
        return int (probaTarget1 > probaTarget0)
    
        
        
class MAP2DClassifier(APrioriClassifier):
    
    def __init__(self,data,arg):
        self.dico = P2D_p(data,arg)
        self.argument = arg
    
    def estimClass(self, attrs):
        #print(attrs)
        value = attrs.get(self.argument)        
        probaTarget1 = self.dico.get(value).get(1)
        probaTarget0 = self.dico.get(value).get(0)
        #print(value,probaTarget1,probaTarget0)
        return int (probaTarget1 > probaTarget0)


def createStringDecomposition(nbRes):
    strDecomposition = ""
    listeNoms = ["o ","ko ","mo ","go ","to "]
    if(nbRes > 1023):   
        nbRes2 = nbRes
        strDecomposition+="= "
        strBuff = ""
        cptL = 0
        while(nbRes2 > 1023):
            strBuff= str(nbRes2%1024)+listeNoms[cptL]+strBuff
            cptL+=1
            nbRes2 = int (nbRes2/1024)
        strDecomposition+= str(nbRes2)+listeNoms[cptL]+strBuff
    return strDecomposition

def nbParams(data,listeArgs=None):
    if not listeArgs:
        listeArgs = list(data.columns.values)
    nbRes = 1
    for i in listeArgs:
        nbRes*=len(np.unique(data.get(i).values))
    nbRes*=8
    strDecomposition = createStringDecomposition(nbRes)
    print( str(len(listeArgs))+" variable(s) : "+str(nbRes)+" octets "+strDecomposition)
    return


def nbParamsIndep(data,listeArgs=None):
    if not listeArgs:
        listeArgs = list(data.columns.values)
    nbRes = 0
    for i in listeArgs:
        nbRes+=len(np.unique(data.get(i).values))
    nbRes*=8
    strDecomposition = createStringDecomposition(nbRes)
    print( str(len(listeArgs))+" variable(s) : "+str(nbRes)+" octets "+strDecomposition)
    return


def drawNaiveBayes(data,arg):
    strBuff = ""   
    for i in data:
        if i != arg:
            strBuff = strBuff + arg + "->" + i +";"
    strBuff = strBuff[:-1]
    return drawGraph(strBuff)

def nbParamsNaiveBayes(data,arg,listeCol=None):
    nbEtatsArg = len(np.unique(data.get(arg).values))
    res = nbEtatsArg
    if listeCol == None:
        listeCol = [i for i in data]
        
    for i in listeCol:
        if i != arg:
            res += len(np.unique(data.get(i).values)) * nbEtatsArg
    res *= 8
    strDecomposition = createStringDecomposition(res)
    print( str(len(listeCol))+" variable(s) : "+str(res)+" octets "+strDecomposition)



class MLNaiveBayesClassifier(APrioriClassifier): 
    def __init__(self,data):
        self.listeCols = [i for i in data]
        dico0 = {}
        dico1 = {}
        for i in data:
            if i != 'target':
                tmp = P2D_l(data,i)
                dico0.update({i : tmp[0]})
                dico1.update({i : tmp[1]})
        self.dico = {0 : dico0, 1 : dico1}
        #self.dico = P2D_l(data,arg)
        self.probaTarget1 = len(data.query('target==1').index) / len(data)


    def estimProbas(self, attrs):
        p1 = 1 #self.probaTarget1
        p0 = 1 #1 - p1
        t1Possible = True
        t0Possible = True
        lenattrs = len(self.listeCols)
        for a in range(lenattrs):
            i = self.listeCols[a]
            if i != 'target':
                if attrs[i] not in list(self.dico[1][i].keys()):
                    t1Possible = False 
                if attrs[i] not in list(self.dico[0][i].keys()):
                    t0Possible = False
                if not(t1Possible and t0Possible):

                    if t1Possible:
                        return {1 : 1.0, 0 : 0.0}
                    return {0 : 1.0, 1 : 0.0}
                p1 *= self.dico[1][i][attrs[i]]
                p0 *= self.dico[0][i][attrs[i]]
        return {0 : p0, 1: p1}

    def estimClass(self,attrs):
        var1 = self.estimProbas(attrs)
        p0 = var1[0]
        p1 = var1[1]
        return (int) ((p1 / (p0 + p1)) > (p0 / (p0 + p1)))

    
    
class MAPNaiveBayesClassifier(APrioriClassifier):
    def __init__(self,data):
        self.listeCols = [i for i in data]
        dico0 = {}
        dico1 = {}
        for i in data:
            if i != 'target':
                tmp = P2D_l(data,i)
                dico0.update({i : tmp[0]})
                dico1.update({i : tmp[1]})
        self.dico = {0 : dico0, 1 : dico1}
        #self.dico = P2D_l(data,arg)
        self.probaTarget1 = len(data.query('target==1').index) / len(data)


    def estimProbas(self, attrs):
        p1 = 1 #self.probaTarget1
        p0 = 1 #1 - p1
        t1Possible = True
        t0Possible = True
        lenattrs = len(self.listeCols)
        for a in range(lenattrs):
            i = self.listeCols[a]
            if i != 'target':
                if attrs[i] not in list(self.dico[1][i].keys()):
                    t1Possible = False 
                if attrs[i] not in list(self.dico[0][i].keys()):
                    t0Possible = False
                if not(t1Possible and t0Possible):

                    if t1Possible:
                        return {1 : 1.0, 0 : 0.0}
                    return {0 : 1.0, 1 : 0.0}
                p1 *= self.dico[1][i][attrs[i]]
                p0 *= self.dico[0][i][attrs[i]]
        p1*= self.probaTarget1
        p0*=(1-self.probaTarget1)
        tmp = p0 + p1
        p0 = p0 / tmp
        p1 = p1 / tmp
        return {0 : p0, 1: p1}

    def estimClass(self,attrs):
        var1 = self.estimProbas(attrs)
        #print(var1)
        p0 = var1[0]
        p1 = var1[1]
        return (int) ((p1 / (p0 + p1)) > (p0 / (p0 + p1)))

import scipy

def isIndepFromTarget(data,attr,seuil):
    valeursPoss = list(np.unique((data.get(attr).values)))
    compteAttr = [[0]*len(valeursPoss),[0]*len(valeursPoss)]
    total = 0
    for i in range(len(data.index)):  
        total+=1
        compteAttr[data.get("target").iloc[i]][valeursPoss.index(data.get(attr).iloc[i])] += 1
    a,b,c,d = scipy.stats.chi2_contingency(compteAttr)
    return b > seuil
    
class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):   
    def __init__(self,data,seuil):
        self.listeCols = []
        listeAttr = [d for d in data]
        for d in listeAttr:
            if not isIndepFromTarget(data,d,seuil):
               self.listeCols.append(d) 
        dico0 = {}
        dico1 = {}
        for i in data:
            if i != 'target':
                tmp = P2D_l(data,i)
                dico0.update({i : tmp[0]})
                dico1.update({i : tmp[1]})
        self.dico = {0 : dico0, 1 : dico1}
        #self.dico = P2D_l(data,arg)
        self.probaTarget1 = len(data.query('target==1').index) / len(data)
    
    def draw(self):
        return drawNaiveBayes(self.listeCols,"target")
    
class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    def __init__(self,data,seuil):
        """
        pour chaque argument, P2D_p(data,arg)

        """
        self.dico = {}
        self.listeCols = []
        listeAttr = [d for d in data]
        for d in listeAttr:
            if not isIndepFromTarget(data,d,seuil):
               self.listeCols.append(d) 
        for i in data:
            if i != 'target' and i in self.listeCols:
                tmp = P2D_p(data,i)
                self.dico.update({i: tmp})
        self.probaTarget1 = len(data.query('target==1').index) / len(data)   
        
    def draw(self):
        return drawNaiveBayes(self.listeCols,"target")   
   
    
def mapClassifiers(dico,data):
    listeX = []
    listeY = []
    for k in dico.keys():
        res = dico[k].statsOnDF(data)
        listeX.append(res['Précision'])
        listeY.append(res['Rappel'])    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    plt.scatter(listeX,listeY)
    i = 1
    for  xy in zip(listeX,listeY):
        ax.annotate(i,xy=xy,textcoords='data')
        i+=1    
    plt.show()
    
    
def MutualInformation(df,x,y):
    acc = 0
    listeX = list(np.unique(df.get(x)))
    listeY = list(np.unique(df.get(y)))
    lendata = len(df.index)
    for i in listeX:
        for j in listeY:
            tmp1 = len(df.query(x+'=='+str (np.asscalar(i))+' and '+y+'=='+str (np.asscalar(j)))) / lendata
            if(tmp1 > 0):
                tmp2 = math.log2(tmp1 / ((len(df.query(x+'=='+str (np.asscalar(i)))) * len(df.query(y+'=='+str (np.asscalar(j)))))/ (lendata**2))) 
                acc+=tmp1 * tmp2
    return acc
 
import time

def ConditionalMutualInformation(df,x,y,z):
    t = time.time()
    listeX = list(np.unique(df.get(x)))
    listeY = list(np.unique(df.get(y)))
    listeZ = list(np.unique(df.get(z)))
    acc = 0 #[[0 for i in range(len(listeY))] for j in range(len(listeX))]
    lendata = len(df.index)
    for i in listeX:
        for j in listeY:
            for k in listeZ:
                tmpx = df[df.get(x) == i]
                tmpxy = tmpx[tmpx.get(y) == j]
                tmpxyz = tmpxy[tmpxy.get(z) == k]
                pxyz = len(tmpxyz) / lendata
                if(pxyz > 0):
                    tmpz = df[df.get(z) == k]
                    tmpxz = tmpz[tmpz.get(x) == i]
                    tmpyz = tmpz[tmpz.get(y) == j]
                    pz = sum([sum([len(tmpz.query(x+'=='+str (np.asscalar(a))+' and '+y+'=='+str (np.asscalar(b)))) for b in listeY]) for a in listeX]) / lendata
                    pxz = sum([len(tmpxz[tmpxz.get(y) == b]) for b in listeY]) / lendata
                    pyz = sum([len(tmpyz[tmpyz.get(x) == a]) for a in listeX]) / lendata
                    acc += pxyz * (math.log2(( pz * pxyz ) / (pxz * pyz)))
    print(time.time() - t)
    return acc
    
def MeanForSymetricWeights(array):
    liste = list(array)
    longueur = len(liste)
    total = 0
    for i in range(longueur):
        for j in range(i+1,longueur):
            total += 2*liste[i][j]
    return total / (longueur**2 - longueur)

def SimplifyConditionalMutualInformationMatrix(array):
    m = MeanForSymetricWeights(array)
    longueur = len(list(array))
    for i in range(longueur):
        for j in range(longueur):
            if array[i][j] < m:
                array[i][j] = 0;
    return array
   
from operator import itemgetter    

def Kruskal(df,array):
    cles = list(df.keys())
    res = []
    longueur = len(list(array))
    for i in range(longueur):
        for j in range(i+1,longueur):
            if array[i][j] > 0:
                res.append((cles[i],cles[j],array[i][j]))
    res = sorted(res,key=itemgetter(2),reverse=True)
    return res[:5]
    
def ConnexStets(liste):
    return 1




