# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:28:05 2016

@author: hplustech
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    plt.imshow(cells, interpolation = 'nearest')
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
    
def cropCell(cells):
    # Check firsts and lasts rows
    while sum(cells[0]) == 0:
        cells.pop(0)
    while sum(cells[-1]) == 0:
        cells.pop()
    
    # Calculate the sum of all the columns
    sumCols=[0]*len(cells[0])
    for n in range(len(cells)):
        for m in range(len(cells[n])):
            sumCols[m]+=cells[n][m]
    
    # Remove columns whose sum is 0
    n=0
    while sumCols[n] == 0:
        n+=1
        for row in cells:
            row.pop(0)
    n=-1
    while sumCols[n] == 0:
        n-=1
        for row in cells:
            row.pop()
            
    return cells
    
def get_generation(cells, generations):
    displayCell(cells)
    # Loop over the generations
    for g in range(generations):
        # Each generation increase the grid size by one in each direction
        cells = expandCell(cells)
        #print(htmlize(cells))
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
        # At the end of each loop shrink the array so that it is cropped around the living cells
        cells = cropCell(cells)
        print(htmlize(cells))
        displayCell(cells)
#    print(htmlize(cells))
    return cells
    


start = np.random.randint(2, size=(8, 10)).tolist()         
get_generation(start,15)