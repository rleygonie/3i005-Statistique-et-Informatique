#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

## Constante
OFFSET = 0.2


class State:
    """ Etat generique d'un jeu de plateau. Le plateau est represente par une matrice de taille NX,NY,
    le joueur courant par 1 ou -1. Une case a 0 correspond a une case libre.
    * next(self,coup) : fait jouer le joueur courant le coup.
    * get_actions(self) : renvoie les coups possibles
    * win(self) : rend 1 si le joueur 1 a gagne, -1 si le joueur 2 a gagne, 0 sinon
    * stop(self) : rend vrai si le jeu est fini.
    * fonction de hashage : renvoie un couple (matrice applatie des cases, joueur courant).
    """
    NX,NY = None,None
    def __init__(self,grid=None,courant=None):
        self.grid = copy.deepcopy(grid) if grid is not None else np.zeros((self.NX,self.NY),dtype="int")
        self.courant = courant or 1
    def next(self,coup):
        pass
    def get_actions(self):
        pass
    def win(self):
        pass
    def stop(self):
        pass
    @classmethod
    def fromHash(cls,hash):
        return cls(np.array([int(i)-1 for i in list(hash[0])],dtype="int").reshape((cls.NX,cls.NY)),hash[1])
    def hash(self):
        return ("".join(str(x+1) for x in self.grid.flat),self.courant)

class Jeu:
    """ Jeu generique, qui prend un etat initial et deux joueurs.
        run(self,draw,pause): permet de joueur une partie, avec ou sans affichage, avec une pause entre chaque coup.
                Rend le joueur qui a gagne et log de la partie a la fin.
        replay(self,log): permet de rejouer un log
    """
    def __init__(self,init_state = None,j1=None,j2=None):
        self.joueurs = {1:j1,-1:j2}
        self.state = copy.deepcopy(init_state)
        self.log = None
    def run(self,draw=False,pause=0.5):
        log = []
        if draw:
            self.init_graph()
        while not self.state.stop():
            coup = self.joueurs[self.state.courant].get_action(self.state)
            log.append((self.state,coup))
            self.state = self.state.next(coup)
            if draw:
                self.draw(self.state.courant*-1,coup)
                plt.pause(pause)
        return self.state.win(),log
    def init_graph(self):
        self._dx,self._dy  = 1./self.state.NX,1./self.state.NY
        self.fig, self.ax = plt.subplots()
        for i in range(self.state.grid.shape[0]):
            for j in range(self.state.grid.shape[1]):
                self.ax.add_patch(patches.Rectangle((i*self._dx,j*self._dy),self._dx,self._dy,\
                        linewidth=1,fill=False,color="black"))
        plt.show(block=False)
    def draw(self,joueur,coup):
        color = "red" if joueur>0 else "blue"
        self.ax.add_patch(patches.Rectangle(((coup[0]+OFFSET)*self._dx,(coup[1]+OFFSET)*self._dy),\
                        self._dx*(1-2*OFFSET),self._dy*(1-2*OFFSET),linewidth=1,fill=True,color=color))
        plt.draw()
    def replay(self,log,pause=0.5):
        self.init_graph()
        for state,coup in log:
            self.draw(state.courant,coup)
            plt.pause(pause)

class MorpionState(State):
    """ Implementation d'un etat du jeu du Morpion. Grille de 3X3.
    """
    NX,NY = 3,3
    def __init__(self,grid=None,courant=None):
        super(MorpionState,self).__init__(grid,courant)
    def next(self,coup):
        state =  MorpionState(self.grid,self.courant)
        state.grid[coup]=self.courant
        state.courant *=-1
        return state
    def get_actions(self):
        return list(zip(*np.where(self.grid==0)))
    def win(self):
        for i in [-1,1]:
            if ((i*self.grid.sum(0))).max()==3 or ((i*self.grid.sum(1))).max()==3 or ((i*self.grid)).trace().max()==3 or ((i*np.fliplr(self.grid))).trace().max()==3: return i
        return 0
    def stop(self):
        return self.win()!=0 or (self.grid==0).sum()==0
    def __repr__(self):
        return str(self.hash())


class Puis4State(State):
    NX,NY = 7,6
    def __init__(self,grid=None,courant=None):
        super(Puis4State,self).__init__(grid,courant)
    def next(self,coup):
        state =  Puis4State(self.grid,self.courant)
        for i in range(len(state.grid)):
            if state.grid[i][coup] !=0:
                state.grid[i-1][coup]=self.courant
                break
            if i == len(state.grid)-1:
                state.grid[i][coup]=self.courant
                break
        state.courant *=-1
        return state
    def get_actions(self):
        res = []
        for i in range(len(self.grid[0])):
            if self.grid[0][i] == 0:
                res.append(i)
        return res
    def win(self):
        for i in range(len(self.grid)-3):
            for j in range(len(self.grid[i])-3):
                #print(i,j)
                t =  self.grid[i][j]
                victoire = False
                if t!= 0:
                    #print(i,j)
                    if (self.grid[i+1][j] == t and  self.grid[i+2][j] == t and  self.grid[i+3][j] == t):
                        victoire = True
                        """
                        self.grid[i][j] = 100*t
                        self.grid[i+1][j] = 100*t
                        self.grid[i+2][j] = 100*t
                        self.grid[i+3][j] = 100*t
                        print(t,"col",self.grid)
                        """
                    if (self.grid[i][j+1] == t and  self.grid[i][j+2] == t and  self.grid[i][j+3] == t):
                        victoire = True
                        """
                        self.grid[i][j] = 100*t
                        self.grid[i][j+1] = 100*t
                        self.grid[i][j+2] = 100*t
                        self.grid[i][j+3] = 100*t
                        print(t,"ligne",self.grid)
                        """
                    if (self.grid[i+1][j+1] == t and  self.grid[i+2][j+2] == t and  self.grid[i+3][j+3] == t):
                        victoire = True
                        """
                        self.grid[i][j] = 100*t
                        self.grid[i+1][j+1] = 100*t
                        self.grid[i+2][j+2] = 100*t
                        self.grid[i+3][j+3] = 100*t
                        print(t,"diago",self.grid)
                        """
                    if victoire:
                        #print(t,i,j,(self.grid[i+1][j] == t and  self.grid[i+2][j] == t and  self.grid[i+3][j] == t),(self.grid[i][j+1] == t and  self.grid[i][j+2] == t and  self.grid[i][j+3] == t),(self.grid[i+1][j+1] == t and  self.grid[i+2][j+2] == t and  self.grid[i+3][j+3] == t))
                        return t
        return 0
    def stop(self):
        return self.win()!=0 or (self.grid==0).sum()==0
    def __repr__(self):
        return str(self.hash())



class Agent:
    """ Classe d'agent generique. Necessite une methode get_action qui renvoie l'action correspondant a l'etat du jeu state"""
    def __init__(self):
        pass

    def get_action(self,state):
        pass

class Agent_alea(Agent):
    def __init__(self):
        pass

    def get_action(self,state):
        #print(state.grid) #[0],state.grid[1],state.grid[2])
        return self.action_alea(state)

    def action_alea(self,state):
        return random.choice(state.get_actions())

def Simulateur(n):
    j1 = Agent_alea()
    j2 = Agent_alea()
    v1 = 0
    v2 = 0
    listeVic1 = []
    listeVic2 = []
    for i in range(1,n+1):
        a = Jeu(Puis4State(),j1,j2)
        a,b = a.run()
        if a == 1:
            v1+=1
        elif a == -1:
            v2+=1
        listeVic1.append((1.0*v1)/i)
        listeVic2.append((1.0*v2)/i)
    plt.subplot()
    plt.plot(listeVic1)
    plt.plot(listeVic2)
    plt.show()
    print(listeVic1[-1],listeVic2[-1])


#Simulateur(100)


class Agent_MonteCarlo(Agent):

    def __init__(self):
        pass

    def get_action(self,state):
        #print(state.grid)
        return self.action_monte_carlos(state)

    def action_monte_carlos(self,state):
        N = 40
        lenstate = len(state.get_actions())
        res = [0.0]*lenstate
        res2 = [0]*lenstate
        if(lenstate > 1):
            for i in range(1,N+1):
                index = random.randint(0,lenstate-1)
                a = state.get_actions()[index]
                nouvelEtat = state.next(a)
                j1tmp = Agent_alea()
                j2tmp = Agent_alea()
                jeutmp = Jeu(nouvelEtat,j1tmp,j2tmp)
                vic,useless = jeutmp.run()
                if state.courant == vic:
                    res[index]+=1
                res2[index]+=1
            for i in range(lenstate):
                if res2[i] != 0:
                    res[i] = res[i] / res2[i]
            return state.get_actions()[res.index(max(res))]
        return state.get_actions()[0]

import time

def Simulateur_bis(n):
    t = time.time()
    j1 = Agent_alea()
    j2 = Agent_MonteCarlo()
    v1 = 0
    v2 = 0
    listeVic1 = []
    listeVic2 = []
    for i in range(1,n+1):
        if(i%3 == 0):
            print(i)
        a = Jeu(Puis4State(),j1,j2)
        a,b = a.run()
        if a == 1:
            v1+=1
        elif a == -1:
            v2+=1
        listeVic1.append((1.0*v1)/i)
        listeVic2.append((1.0*v2)/i)
    plt.subplot()
    plt.plot(listeVic1)
    plt.plot(listeVic2)
    #plt.plot([1 - listeVic1[x] - listeVic2[x] for x in range(len(listeVic1))])
    print(listeVic1[-1],listeVic2[-1], 1 - listeVic1[-1] - listeVic2[-1])
    print( time.time() - t)
    plt.show()


Simulateur_bis(200)

"""
PARTIE 3
"""


class Agent_UCT(Agent):

    def __init__(self):
        self.N = 0
        self.Noeud = None
        self.Noeud2 = None


    def get_action(self,state):
        self.N = 40
        #print(state.grid)
        lens = len(state.get_actions())
        #self.N -= lens
        joueur = state.courant
        self.Noeud = Noeud_UCT()
        self.Noeud2 = self.Noeud
        Noeud = self.Noeud
        Noeud = self.init_tree(state,Noeud,joueur)
        for i in range(self.N):
            self.play_ucb(state, self.Noeud,joueur)
        ltmp = []
        for i in range(lens):
            ltmp.append((1.0*Noeud.enfants[i][1].nbVic) / Noeud.enfants[i][1].nbAppel)
        return state.get_actions()[ltmp.index(max(ltmp))]

    def state_in(self,state,liste_noeud):
        for i in liste_noeud:
            if i[0] == state:
                return True
        return False

    def init_tree(self,state,noeud,joueur):
        lenstate = len(state.get_actions())
        Noeud_Pere = noeud

        for i in range(lenstate):
            self.N -= 1
            nouvelEtat = state.next(state.get_actions()[i])
            j1tmp = Agent_alea()
            j2tmp = Agent_alea()
            jeutmp = Jeu(nouvelEtat,j1tmp,j2tmp)
            vic,useless = jeutmp.run()
            if joueur == vic:
                vic = 1
            else:
                vic = 0
            if not self.state_in(state,Noeud_Pere.enfants):
                Noeud_Pere.ajoute_fils(nouvelEtat,Noeud_UCT(Noeud_Pere))
            for i in Noeud_Pere.enfants:
                if i[0].__repr__()[2:11] == nouvelEtat.__repr__()[2:11]:
                    i[1].appel(vic)
                    break

        return  Noeud_Pere

    def play_ucb(self, state, Noeud_Pere,joueur):
        #print(state,Noeud_Pere)
        #while(self.Noeud2.pere):
        #    self.Noeud2 = self.Noeud2.pere
        #print([self.Noeud2.enfants[i][1].nbVic for i in range(len(self.Noeud2.enfants))])
        if Noeud_Pere.estFeuille():
            a = self.init_tree(state,Noeud_Pere,joueur)
            #print("refdugidhsjtiuhjniustryjuiui",a)
            return a
        else:
            #print([Noeud_Pere.enfants[i][1].nbVic for i in range(len(Noeud_Pere.enfants))])
            tmpListe = []
            for k in range(len(Noeud_Pere.enfants)):
                i = Noeud_Pere.enfants[k]
                tmpListe.append([i[0],(1.0*(i[1].nbVic))/i[1].nbAppel])
                tmpListe[-1][1]+= np.sqrt((2.0*np.log10(Noeud_Pere.nbAppel))/max(1,i[1].nbAppel))
            cpt = 0
            i = 0
            for k in range(len(tmpListe)):
                #print(tmpListe[k][1] , cpt,tmpListe[k][1] > cpt)
                if tmpListe[k][1] > cpt:
                    cpt = tmpListe[k][1]
                    i = k
            #print(i)
            nouvelEtat = state.next(state.get_actions()[i])
            #print(state,nouvelEtat)
            for i in Noeud_Pere.enfants:
                #print(i[0].__repr__()[2:11], nouvelEtat.__repr__()[2:11], i[0].__repr__()[2:11] == nouvelEtat.__repr__()[2:11] )
                if i[0].__repr__()[2:11] == nouvelEtat.__repr__()[2:11]:
                    #print("0")
                    self.Noeud = i[1]
                    a = self.play_ucb(nouvelEtat,i[1],joueur)
                    #print(a)
                    return a


class Noeud_UCT:

    def __init__(self,Pere=None):
        self.nbVic = 0
        self.nbAppel = 0
        self.enfants = []
        self.pere = Pere

    def appel(self,victoire):
        self.nbVic += victoire
        self.nbAppel += 1
        if self.pere:
            self.pere.appel(victoire)

    def ajoute_fils(self,state,fils):
        self.enfants.append([state,fils])

    def estFeuille(self):
        return self.enfants == []


def Simulateur_uct(n):
    t = time.time()
    j2 = Agent_UCT()
    j1 = Agent_MonteCarlo()
    v1 = 0
    v2 = 0
    listeVic1 = []
    listeVic2 = []
    for i in range(1,n+1):
        a = Jeu(MorpionState(),j1,j2)
        a,b = a.run()
        if a == 1:
            v1+=1
        elif a == -1:
            v2+=1
        listeVic1.append((1.0*v1)/i)
        listeVic2.append((1.0*v2)/i)
    plt.subplot()
    plt.plot(listeVic1)
    plt.plot(listeVic2)
    print( time.time() - t)
    plt.show()
    print(listeVic1[-1],listeVic2[-1])


#Simulateur_uct(250)












