# Code to run the evolutionary simulation from the MA10247 Foundations and Connections Lecture
# Code written by Cameron Smith
# Feb 2023

#%% Packages to import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import pickle
import os
import imageio

#%% Parameter values

# Host demographics
a = 1
b = 0.25
q = 0.1

# Infection parameters
betaMin = 0
betaMax = 2
nBeta = 51
betaVec = np.linspace(betaMin, betaMax, nBeta)
gamma = 0.5

# Virulence function
alpha = lambda beta: beta**2

# Initial conditions
initBetaInd = 20 # Initial transmission index
initBeta = betaVec[initBetaInd]  # Initial transmission
initState = np.zeros(nBeta+1)
initState[0] = 1
initState[initBetaInd+1] = 0.01

# Times
tEco = 100  # Final ecological time
dtEco = 0.1  # Time-step
tEcoVec = np.arange(0, tEco+dtEco, dtEco)
ntEvo = 201  # Number of evolutionary steps

# %% Functions

# ODE RHS function
def ODEFun(state, t, b, a, inds=None):
    '''
    Function for the RHS of the ODEs.
    state are the current values of the state variables
    t is the current time
    b is the background mortality
    a is the birth rate in the absence of competition
    inds are the indices for the pathogens in circulation. If none, returns all ODEs, if specified, only returs ODEs for the indices given.
    '''

    # Indices if None specified
    if inds == None:
        inds = [ii for ii in range(nBeta)]

    # Initialise the RHS
    RHS = np.zeros(len(inds)+1)

    # Susceptibles
    RHS[0] = sum(state)*(a - q*sum(state)) - b*state[0] - sum(betaVec[inds]*state[1:])*state[0] + gamma*sum(state[1:])
    
    # Infected individuals
    RHS[1:] = betaVec[inds]*state[1:]*state[0] - b*state[1:] - gamma*state[1:] - alpha(betaVec[inds])*state[1:]

    # Return the RHS
    return(RHS)

# ODE step
def ODEStep(state, t, dt, b, a, inds = None):
    '''
    Use the implicit Euler method to complete an ODE solution step
    state are the current values of the state variables
    t is the current time
    dt is the time-step
    b is the background mortality
    a is the birth rate in the absence of competition
    inds are the indices for the pathogens in circulation. If none, returns all ODEs, if specified, only returs ODEs for the indices given.
    '''

    # Sort the indices
    if inds == None:
        inds = [ii for ii in range(nBeta)]

    # State indices need to also include the susceptible group
    stateInds = [[-1], inds]
    stateInds = [item+1 for sublist in stateInds for item in sublist]

    # Calculate the truncated new State
    newState = root(lambda X: X - ODEFun(X, t, b, a, inds)*dt - state[stateInds], state[stateInds]).x

    # Embed into a state variable of the correct size
    nS = np.zeros(len(state))
    nS[stateInds] = newState
    return(nS)

def ODESim(initState, finalTime, dt, b, a, inds = None, plot=False):
    '''
    Solve the ODEs until the Final time
    initState is the initial state variable
    finalTime is the end of the simulation
    dt is the time-step
    b is the background mortality
    a is the birth rate in the absence of competition
    inds are the indices for the pathogens in circulation. If none, returns all ODEs, if specified, only returs ODEs for the indices given.
    Setting plot to True will output the ecological dynamics as a plot
    '''

    # Find the time vector
    tVec = np.arange(0, finalTime+dt, dt)
    
    # Initialise the state vector
    stateBef = initState
    stateStore = np.zeros((len(tVec), nBeta+1))
    stateStore[0,:] = stateBef

    # Loop through the time vector and use ODEStep
    for ind, t in enumerate(tVec):

        # Find the updated state
        stateAfter = ODEStep(stateBef, t, dt, b, a, inds=inds)
        stateBef = stateAfter
        stateStore[ind,:] = stateBef

    # If plotting, create the figure and display
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(tVec, stateStore[:,0], 'b', lw=2, label='S')
        ax.plot(tVec, stateStore[:,initBetaInd+1], 'r--', lw=2)

        ax.set_xlabel(r'Time in days, $t$')
        ax.set_ylabel('Host density')

        plt.show()

    # Return the final state
    return(stateAfter)

#%% Evolutionary simulation

def evoSim(initState, initBetaInd, b, a):
    '''
    Code to conduct an evolutionary simulation for a specified background mortality and birth rate
    initState is the initial state variable
    finalTime is the end of the simulation
    b is the background mortality
    a is the birth rate in the absence of competition
    '''

    # Create storage matrices
    stateStore = np.zeros((ntEvo, nBeta+1))
    indStore = np.zeros((ntEvo, nBeta))

    # Initialise
    currentState = initState
    currentInds = [initBetaInd]

    # Loop through evolutionary time
    for m in range(0, ntEvo):

        # Store before
        stateStore[m, :] = currentState
        tempInds = np.zeros(nBeta, dtype=int)
        tempInds[currentInds] = 1
        indStore[m, :] = tempInds
        del tempInds

        # Update the population-level dynamics to steady state
        newState = ODESim(currentState, tEco, dtEco, b, a, inds = currentInds)

        # Check for any extinction events
        newInds = []
        for ind, val in enumerate(currentInds):

            # Check if the index is still present
            if newState[val+1] > 1e-5:
                newInds.append(val)
            else:
                currentState[val+1] = 0

        # Mutate based on what is left
        if len(newInds) == 1:
            mutInd = newInds[0]
        elif len(newInds) == 0:
            print('All pathogens extinct after ' + str(m) + ' steps')
            return(stateStore, indStore)
        else:
            sumState = sum(newState[newInds])
            randSumState = np.random.rand()*sumState
            jj = 0
            cumSum = newState[newInds][0]
            if randSumState > cumSum:
                jj += 1
                cumSum += newState[newInds][jj]
            mutInd = newInds[jj]

        # Mutate
        if mutInd == 0:
            newInd = 1
        elif mutInd == nBeta:
            newInd = nBeta - 1
        else:
            newInd = (mutInd + np.sign(np.random.rand()-0.5)).astype(int)
        newState[newInd+1] += 0.01

        # Add the new index to the list and initialise
        newInds.append(newInd)
        
        # Remove any duplicates and order
        newInds = sorted(list(np.unique(newInds)))
        
        # Update the current states
        currentState = newState
        currentInds = newInds

    return(stateStore, indStore)

# Function to run over a vector of background mortality values
def mortSweep(initState, initBetaInd, bVec, a, saveDir='./code/bSweep/'):
    '''
    Code that conducts an evolution simulation for every background mortality value in bVec
    initState is the initial state variable
    finalTime is the end of the simulation
    bVec is the vector of background mortality
    a is the birth rate in the absence of competition
    saveDir is the directory to save the data in
    '''

    # Storage tensors
    stateStorage = np.zeros((ntEvo, nBeta+1, len(bVec)))
    indStorage = np.zeros((ntEvo, nBeta, len(bVec)))

    # Loop through the b vector
    for bInd, b in enumerate(bVec):

        # Evo sim
        SS, II = evoSim(initState, initBetaInd, b, a)

        # Store
        stateStorage[:, :, bInd] = SS
        indStorage[:, :, bInd] = II

    # Save the data
    pdict = {'state': stateStorage, 'inds': indStorage, 'bVec': bVec, 'a': a}
    file = open(saveDir + 'data.pkl', 'wb')
    pickle.dump(pdict, file)
    file.close()

# Function to run over a vector of birth rate values
def birthSweep(initState, initBetaInd, b, aVec, saveDir='./code/aSweep/'):
    '''
    Code that conducts an evolution simulation for every background mortality value in bVec
    initState is the initial state variable
    finalTime is the end of the simulation
    b is the background mortality
    aVec is the vector of birth rate in the absence of competition
    saveDir is the directory to save the data in
    '''

    # Storage tensors
    stateStorage = np.zeros((ntEvo, nBeta+1, len(aVec)))
    indStorage = np.zeros((ntEvo, nBeta, len(aVec)))

    # Loop through the b vector
    for aInd, a in enumerate(aVec):

        # Evo sim
        SS, II = evoSim(initState, initBetaInd, b, a)

        # Store
        stateStorage[:, :, aInd] = SS
        indStorage[:, :, aInd] = II

    # Save the data
    pdict = {'state': stateStorage, 'inds': indStorage, 'aVec': aVec, 'b': b}
    file = open(saveDir + 'data.pkl', 'wb')
    pickle.dump(pdict, file)
    file.close()

#%% Plotting code

def plotMortSweep(dataFile):
    '''
    Code to create a gif over the mortality parameters, as well as some snapshot figures.
    Gif will be used in the actual lecture and on online material (HTML?)
    Static image will be for the pdf
    dataFile should have the .pkl extension
    '''

    # Load the data
    file = open(dataFile, 'rb')
    pdict = pickle.load(file)
    file.close()

    # Extract the data
    stateStorage = pdict['state']
    indStorage = pdict['inds']
    bVec = pdict['bVec']
    a = pdict['a']
    del pdict

    # Configure settings for plotting
    figWidth = 8
    figHeight = 12
    fsize = 18  # Fontsize
    plt.rcParams['font.size'] = fsize

    # Loop through the bVec
    for bInd, b in enumerate(bVec):
        
        # Create 5 frames per value to slow down the gif
        for ii in range(5):

            # Initialise the figure
            fig = plt.figure(figsize=(figWidth, figHeight))
            ax = fig.add_subplot(111)
            plt.rcParams['font.size'] = fsize

            # Plot using pcolormesh
            BB, TT = np.meshgrid(betaVec, np.arange(ntEvo))
            mask = indStorage[:, :, bInd].astype(int)
            scatterPairs = np.array([BB[mask==1], TT[mask==1], stateStorage[:, 1:, bInd][mask==1]]).transpose()
            ax.scatter(scatterPairs[:,0], scatterPairs[:,1], c=scatterPairs[:,2], cmap='Greys', label=r'b = %.2f' % bVec[bInd])
            ax.set_xlabel(r'Transmission, $\beta$')
            ax.set_xlim(betaVec[0], betaVec[-1])
            ax.set_ylabel('Evolutionary time')
            ax.set_ylim(0, ntEvo-1)
            ax.set_title(r'Background mortality $b$ = %.2f' % b)

            # Save the figure and close
            plt.savefig('./code/bSweep/Figs/' + str(5*bInd + ii).zfill(4) + '.png')
            fig.clf()
            plt.close()
    
    images = []
    for file_name in sorted(os.listdir('./code/bSweep/Figs/')):
        if file_name.endswith('.png'):
            file_path = os.path.join('./code/bSweep/Figs/', file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('./img/bSweep.gif', images)

    # Create a static version
    bVals = [0,(len(bVec)-1)//2,-1]

    figStat = plt.figure(figsize=(figWidth, figHeight))
    axStat = figStat.add_subplot(111)

    for index, bInd in enumerate(bVals):
        cols = 'r'*(index==0) + 'm'*(index==1) + 'b'*(index==2)
        styles = '-'*(index==0) + ':'*(index==1) + '--'*(index==2)
        BB, TT = np.meshgrid(betaVec, np.arange(ntEvo))
        mask = indStorage[:, :, bInd].astype(int)
        ave = np.sum(BB*stateStorage[:,1:,bInd]*mask, axis=1)/np.sum(stateStorage[:,1:,bInd]*mask,axis=1)
        axStat.plot(ave, np.arange(ntEvo), c=cols, ls=styles, lw=2, label=r'b = %.2f' % bVec[bInd])
        axStat.plot(ave[-1], 0, c=cols, marker='o', ms=10)

    axStat.set_xlim(0,2)
    axStat.set_xlabel(r'Transmission, $\beta$')
    axStat.set_ylim(0, ntEvo-1)
    axStat.set_ylabel('Evolutionary time')

    plt.legend()

    plt.savefig('./img/bSweepStatic.png')

def plotBirthSweep(dataFile):
    '''
    Code to create a gif over the birth parameters, as well as some snapshot figures.
    Gif will be used in the actual lecture and on online material (HTML?)
    Static image will be for the pdf
    dataFile should have the .pkl extension
    '''

    # Load the data
    file = open(dataFile, 'rb')
    pdict = pickle.load(file)
    file.close()

    # Extract the data
    stateStorage = pdict['state']
    indStorage = pdict['inds']
    b = pdict['b']
    aVec = pdict['aVec']
    del pdict

    # Configure settings for plotting
    figWidth = 8
    figHeight = 12
    fsize = 18  # Fontsize
    plt.rcParams['font.size'] = fsize

    # Loop through the bVec
    for aInd, a in enumerate(aVec):
        
        # Create 5 frames per value to slow down the gif
        for ii in range(5):

            # Initialise the figure
            fig = plt.figure(figsize=(figWidth, figHeight))
            ax = fig.add_subplot(111)
            plt.rcParams['font.size'] = fsize

            # Plot using pcolormesh
            BB, TT = np.meshgrid(betaVec, np.arange(ntEvo))
            mask = indStorage[:, :, aInd].astype(int)
            scatterPairs = np.array([BB[mask==1], TT[mask==1], stateStorage[:, 1:, aInd][mask==1]]).transpose()
            ax.scatter(scatterPairs[:,0], scatterPairs[:,1], c=scatterPairs[:,2], cmap='Greys', label=r'a = %.2f' % aVec[aInd])
            ax.set_xlabel(r'Transmission, $\beta$')
            ax.set_xlim(betaVec[0], betaVec[-1])
            ax.set_ylabel('Evolutionary time')
            ax.set_ylim(0, ntEvo-1)
            ax.set_title(r'Birth rate $a$ = %.2f' % a)

            # Save the figure and close
            plt.savefig('./code/aSweep/Figs/' + str(5*aInd + ii).zfill(4) + '.png')
            fig.clf()
            plt.close()
    
    images = []
    for file_name in sorted(os.listdir('./code/aSweep/Figs/')):
        if file_name.endswith('.png'):
            file_path = os.path.join('./code/aSweep/Figs/', file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('./img/aSweep.gif', images)

    # Create a static version
    aVals = [0,(len(aVec)-1)//2,-1]

    figStat = plt.figure(figsize=(figWidth, figHeight))
    axStat = figStat.add_subplot(111)

    for index, aInd in enumerate(aVals):
        cols = 'r'*(index==0) + 'm'*(index==1) + 'b'*(index==2)
        styles = '-'*(index==0) + ':'*(index==1) + '--'*(index==2)
        BB, TT = np.meshgrid(betaVec, np.arange(ntEvo))
        mask = indStorage[:, :, aInd].astype(int)
        ave = np.sum(BB*stateStorage[:,1:,aInd]*mask, axis=1)/np.sum(stateStorage[:,1:,aInd]*mask,axis=1)
        axStat.plot(ave, np.arange(ntEvo), c=cols, ls=styles, lw=2, label=r'a = %.2f' % aVec[aInd])
        axStat.plot(ave[-1], 0, c=cols, marker='o', ms=10)

    axStat.set_xlim(0,2)
    axStat.set_xlabel(r'Transmission, $\beta$')
    axStat.set_ylim(0, ntEvo-1)
    axStat.set_ylabel('Evolutionary time')

    plt.legend()

    plt.savefig('./img/aSweepStatic.png')

# mortSweep(initState, initBetaInd, np.linspace(0,0.5,21), a)
plotMortSweep('./code/bSweep/data.pkl')

# birthSweep(initState, initBetaInd, b, np.linspace(0.5, 1.5, 21))
plotBirthSweep('./code/aSweep/data.pkl')

# print(ODESim(initState, tEco, dtEco, b, a, inds = [initBetaInd], plot=True)[[0, initBetaInd+1]])