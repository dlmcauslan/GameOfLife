# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:28:05 2016

@author: hplustech
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation

def htmlize(array):
    s = []
    for row in array:
        for cell in row:
            s.append('▓▓' if cell else '░░')
        s.append('\n')
    return ''.join(s)
    
def displayCell(cells):
    fig1 = plt.figure(1)
    fig1.clf()
    plt.imshow(cells, interpolation = 'nearest', cmap = "Greys")
    fig1.show()
    
    
def countNeighbours(cells, n, m):
    numNeighbours = 0
    neighbours = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    nDim = len(cells)
    mDim = len(cells[0])
    # Loop over neighbours
    for pos in neighbours:
        newN = n+pos[0]
        newM = m+pos[1]
        if newN>=0 and newN<nDim and newM>=0 and newM<mDim and cells[newN][newM] == 1:
            numNeighbours+=1       
    return numNeighbours
 
   
def expandCell(cells):
    # First row of 0s
    newCells = [[0]*(len(cells[0])+2)]
    # Add 0s at the front and end of each row
    for row in cells:
        newRow = [0]+row+[0]
        newCells.append(newRow)
    # Add final row of 0s
    newCells.append([0]*(len(cells[0])+2))
    return newCells
 
   
def cropCell(cells, expandDims):
    # Add one to each of expandDims
    for i in range(len(expandDims)):
        expandDims[i]+=1
    # Check firsts and lasts rows
    while sum(cells[0]) == 0:
        cells.pop(0)
        expandDims[3]-=1
    while sum(cells[-1]) == 0:
        cells.pop()
        expandDims[2]-=1   
    # Calculate the sum of all the columns
    sumCols=[0]*len(cells[0])
    for n in range(len(cells)):
        for m in range(len(cells[n])):
            sumCols[m]+=cells[n][m]    
    # Remove columns whose sum is 0
    n=0
    while sumCols[n] == 0:
        n+=1
        expandDims[1]-=1
        for row in cells:
            row.pop(0)
    n=-1
    while sumCols[n] == 0:
        n-=1
        expandDims[0]-=1
        for row in cells:
            row.pop()           
    return cells, expandDims


def cellLooper(cells):
    # Need to copy the cells because all births/deaths happen simultaneously
    currGen = copy.deepcopy(cells)
    # Loop over every cell in the grid, determine whether it is alive, then count its neighbours to decide 
    # the action to perform
    for n in range(len(cells)):
        for m in range(len(cells[0])):
            # Count number of neighbours
            numNeighbours = countNeighbours(currGen, n, m)
            # Is cell currently alive
            if currGen[n][m] == 1:                
                # Choose action to perform based on number of neighbours
                if numNeighbours < 2 or numNeighbours > 3:
                    cells[n][m] = 0
            else:
                # Does cell reproduce?
                if numNeighbours == 3:
                    cells[n][m] = 1
    return cells

    
def findMaxDims(cells, generations):
    expandDims =[0, 0, 0, 0]       # X+, X-, Y+, Y-
    maxExpand = [0, 0, 0, 0]       # X+, X-, Y+, Y-
    # Loop over the generations
    for g in range(generations):
        # Each generation increase the grid size by one in each direction
        cells = expandCell(cells)
        # Loop over all the cells
        cells = cellLooper(cells)
        # At the end of each loop shrink the array so that it is cropped around the living cells
        cells, expandDims = cropCell(cells, expandDims)
        # Then update maxExpand
        for i in range(len(expandDims)):
            maxExpand[i] = max(maxExpand[i], expandDims[i])
#        print(htmlize(cells))
    return maxExpand

    
def getGeneration(cells, generations):
#    displayCell(cells)
    totalGens = np.zeros((len(cells), len(cells[0]), generations+1))
    totalGens[:,:,1] = cells
    # Loop over the generations
    for g in range(generations):
        # Loop over all the cells
        cells = cellLooper(cells)
        totalGens[:,:,g+1]=cells
    print(htmlize(cells))
#        displayCell(cells)
    return totalGens
    

# Overall function that runs the total simulation    
def runSim(cells, generations):
    print(htmlize(cells))
    # Run once to find the maximum grid size required
    maxExpand = findMaxDims(cells,generations)
    # Create correct sized grid with cells in the right place
    newCells = np.zeros((len(cells)+maxExpand[2]+maxExpand[3],len(cells[0])+maxExpand[0]+maxExpand[1]))
    rows = maxExpand[3] + np.array(range(len(cells)))
    cols = maxExpand[1] + np.array(range(len(cells[0])))
    newCells[rows[:, np.newaxis],cols] = cells
    # Run again with constant grid size
    totalGens = getGeneration(newCells,generations)
    return totalGens
    
    

start = np.random.randint(2, size=(15, 25)).tolist()
generations = 100        
totalGens = runSim(start,generations)

#%%
fig = plt.figure(2)
fig.clf()

def init():
    plt.title("Generation = {}".format(1))
    plt.imshow(totalGens[:,:,1], interpolation = 'nearest', cmap = "Greys")
    
def animate(i):
    plt.title("Generation = {}".format(i))
    plt.imshow(totalGens[:,:,i], interpolation = 'nearest', cmap = "Greys")
    
anim = animation.FuncAnimation(fig, animate, init_func=init, frames= generations+1, interval=500)

#anim.save('GameOfLifeNew.mp4', fps=2)

plt.show()
