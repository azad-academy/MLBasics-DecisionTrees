'''
Azad-Academy
Author: J. Rafid Siddiqui
jrs@azaditech.com
https://www.azaditech.com

'''

import sys
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import math
from itertools import permutations
import matplotlib

import sklearn
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from IPython.display import display


def plot_data(X,Y,model=None,canvas=None,xtitle=None,ytitle=None,colors=None,plt_title=None,color_map=plt.cm.RdBu):
            
    if(colors is None):
        colors = np.random.rand(max(Y)+1,3)    
        
    if(canvas is None):
        fig, ax = plt.subplots(figsize=(11,8))
    else:
        ax = canvas
        ax.cla()
    
    if(plt_title is not None):
        ax.set_title(plt_title)  
    
    if(model is not None):  #Plotting the decision boundary
        h = .05 #mesh grid resolution
        
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=color_map, alpha=.8)
    
    if(X.shape[1]>2):
        ax.scatter3D(X[:,0],X[:,1],X[:,2],color=np.array(colors)[Y],alpha=0.6)  #plotting the 3D points
        ax.grid(False)
    else:
        ax.scatter(X[:,0],X[:,1],color=np.array(colors)[Y],alpha=0.6)  #plotting the 2D points
            
    if(xtitle is not None):
        ax.set_xlabel(xtitle,fontweight='bold',fontsize=16)
    
    if(xtitle is not None):
        ax.set_ylabel(ytitle,fontweight='bold',fontsize=16)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

       
def entropy(Y):
    
    clabels = np.unique(Y)
    entropy = 0
    for label in clabels:
        p = len(Y[Y == label]) / len(Y)
        entropy += -p * np.log2(p)
    return entropy

def gini_index(Y):
            
    clabels = np.unique(Y)
    gini = 0
    for label in clabels:
        p = len(Y[Y == label]) / len(Y)
        gini += p**2
    return 1 - gini


def get_subscript(x):
    x = str(x)
    n = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(n), ''.join(s))
    return x.translate(res)

def get_supscript(x):
    x = str(x)
    n = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(n), ''.join(s))
    return x.translate(res)


def height(root):
    return 1 + max(height(root.left), height(root.right)) if root else -1  

def DrawTree(root,scale=1):
    
    nlevels = height(root)
    width =  int(scale)*pow(2,nlevels+1)

    q=[(root,0,width,'c')]
    levels=[]

    while(q):
        node,level,x,align= q.pop(0)
        if node:            
            if len(levels)<=level:
                levels.append([])
        
            levels[level].append([node,level,x,align])
            seg= width//(pow(2,level+1))
            q.append((node.left,level+1,x-seg,'l'))
            q.append((node.right,level+1,x+seg,'r'))

    for i,l in enumerate(levels):
        pre=0
        preline=0
        linestr=''
        pstr=''
        seg= width//(pow(2,i+1))
        for n in l:
            valstr= str(n[0].text)
            if n[3]=='r':
                linestr+=' '*(n[2]-preline-1-seg-seg//2)+ '¯'*(seg +seg//2)+'\\'
                preline = n[2] 
            if n[3]=='l':
               linestr+=' '*(n[2]-preline-1)+'/' + '¯'*(seg+seg//2)  
               preline = n[2] + seg + seg//2
            pstr+=' '*(n[2]-pre-len(valstr))+valstr 
            pre = n[2]
        print(linestr)
        print(pstr) 