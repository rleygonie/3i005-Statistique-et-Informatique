import random
import math
import numpy as np

TAILLEMACHINE = 20

def gainAction(listeProba,numLevier):
    """
    list,int -> boolean
    renvoie un gain boolean
    """
    if(listeProba[numLevier] > random.random()):
        return 1
    return 0


def genereMachine(nbLeviers):
    return [random.random() for x in range(nbLeviers)]


machineSimul = genereMachine(TAILLEMACHINE)


def algoAlea(listeStats,nbJoues):
    i = random.randint(0,len(listeStats)-1)
    nbJoues[i]+=1
    listeStats[i] += gainAction(machineSimul,i)
    return i
    

def algoGreedy(listeStats,nbJoues):
    if (sum(nbJoues) < len(listeStats)*50):
        return algoAlea(listeStats,nbJoues)
    tmp = []
    for i in range(len(listeStats)):
        if(nbJoues[i] == 0):
            tmp.append(1)
        else:
            tmp.append((1.0*listeStats[i])/nbJoues[i])
    joue = tmp.index(max(tmp))
    nbJoues[joue]+=1
    listeStats[joue] += gainAction(machineSimul,joue)
    return joue
    
def algoEpsGreedy(listeStats,nbJoues):
    epsilon= 0.01
    if (random.random() > epsilon):
        return algoGreedy(listeStats,nbJoues)
    else:
        return algoAlea(listeStats,nbJoues)
    
def algoUCB(listeStats,nbJoues):
    tmp = []
    for i in range(len(listeStats)):
        if(nbJoues[i] == 0):
            tmp.append(1)
        else:
            tmp.append((1.0*listeStats[i])/nbJoues[i])
        if(sum(nbJoues)):
            tmp[i]+= np.sqrt((2*np.log10(sum(nbJoues)))/max(1,nbJoues[i]))
    joue = np.argmax(np.array(tmp))
    nbJoues[joue]+=1
    listeStats[joue] += gainAction(machineSimul,joue)    
    return joue
    

def testAlea(tailleMachine=TAILLEMACHINE):
    stats = [0]*tailleMachine
    nbCoups = [0]*tailleMachine
    for i in range(100000):
        algoAlea(stats,nbCoups)
    for i in range(tailleMachine):
        tmp1 =machineSimul[i]
        tmp2 =(1.0*stats[i])/nbCoups[i]
        print(tmp1,tmp2,abs(tmp1-tmp2))
    return
#testAlea()

def testGreedy(taillemachine=TAILLEMACHINE):
    stats= [0]*taillemachine
    nbCoups=[0]*taillemachine
    for i in range (100000):
        algoGreedy(stats,nbCoups)
    for i in range(taillemachine):
        tmp1 =machineSimul[i]
        tmp2 =(1.0*stats[i])/nbCoups[i]
        print(tmp1,tmp2,abs(tmp1-tmp2),nbCoups[i])
    return

#testGreedy()

def testEpsGreedy(taillemachine=TAILLEMACHINE):
    stats= [0]*taillemachine
    nbCoups=[0]*taillemachine
    for i in range (100000):
        algoEpsGreedy(stats,nbCoups)
    for i in range(taillemachine):
        tmp1 =machineSimul[i]
        tmp2 =(1.0*stats[i])/nbCoups[i]        
        print(tmp1,tmp2,abs(tmp1-tmp2),nbCoups[i])
    return

#testEpsGreedy()

def testUCB(taillemachine=TAILLEMACHINE):
    stats= [0]*taillemachine
    nbCoups=[0]*taillemachine
    for i in range (100000):
        algoUCB(stats,nbCoups)
    for i in range(taillemachine):
        tmp1 =machineSimul[i]
        if(nbCoups[i]):
            tmp2 =(1.0*stats[i])/nbCoups[i]   
        else:
            tmp2 = 0
        print(tmp1,tmp2,abs(tmp1-tmp2),nbCoups[i])
    return

testUCB()






