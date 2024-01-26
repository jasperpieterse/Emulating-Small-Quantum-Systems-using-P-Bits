import numpy as np
import random
import os
import matplotlib.pyplot as plt
from itertools import product
import math
from tqdm import tqdm
import sys

#TODO: remove handle bit and reduce the number of auxiliary bits. 

# Global variables
random.seed(123) # setting the seed of rng for reproducibility
NM = 8 # number of p-bits in the system

# initial state of the spins (randomly chosen)
m = np.array([-1, 1, -1])

# h vector (a column vector)
h = np.zeros((NM, 1))

Look = [2**i for i in range(NM-1, -1, -1)] # bin identifier vector

NT = 10000  # Number of samples (10000 might be too little, since this results in some variability between the exact and probablistic values)
beta = 1    # inverse temperature = 1/kT, where T is absolute temperature and k is Boltzmann constant
# higher value for beta makes the probable states more probable and the non-probable states even less probable

E = np.zeros(NT)

def calculate_J(truth_table, nr_aux_bits, nr_bits):
    U = truth_table
    J_width = nr_aux_bits+nr_bits
    count = 0

    for vals in product([0, 1], repeat=nr_bits*nr_aux_bits):
        # get new configuration of binary array
        arr = np.array(vals).reshape((nr_aux_bits, nr_bits))

        # tranform zeros into -1
        for i in range(nr_aux_bits):
            for j in range(nr_bits):
                if arr[i][j] == 0:
                    arr[i][j] = -1

        # print("shape of arr: ", arr.T.shape)
        # print("shape of U: ", U.T.shape)

        # concatinate the random array with U
        new_U = np.concatenate((arr.T, U.T)).T
        UU = np.dot(new_U, new_U.T)
        # print("shape of new_U: ", new_U.shape)

        # see if U*UT is an identity matrix
        is_id = True
        for i in range(nr_aux_bits):
            for j in range(nr_aux_bits):
                if UU[i][j] != 0 and i != j:
                    is_id = False

        # if it is an identity matrix then save and break
        if is_id:
            count += 1
            if count >= 2: # for some reason not all of them work, but this one does? 2, 3, 4, 5, 6, 7, 10, the're all slightly different but they all work.
                U = new_U
                break

    J = np.zeros((J_width,J_width))
    for i in range(nr_aux_bits):
        Ui = np.array([U[i]])
        # print("shape of Ui: ", Ui.shape)
        J += np.dot(Ui.T, Ui).T

    J = J/2

    # put zeros in the diagonal of J
    for i in range(J_width):
        for j in range(J_width):
            if i == j:
                J[i][j] = 0

    return J


def exact_simulation(J):
    Pe = np.zeros(2**NM)
    h = np.zeros((J.shape[0], 1))
    Look = [2**i for i in range(J.shape[0]-1, -1, -1)] # bin identifier vector

    for ii in range(0, 2**NM): # each of the NM variables has two states, thus the system has 2^NM states in total
        # look up the documentation of de2bi function
        # generates a state of the spins
        binary_str = np.binary_repr(ii, width=NM)
        m = (2 * np.array(list(binary_str), dtype=np.int8) - 1) # scaling to -1 and 1.

        # compute the energy of the system for the given states of spins
        E[ii] = -0.5 * m.T @ J @ m + h.T @ m

        # probability of given state is proportional to exp(-beta*E)
        Pe[ii] = np.exp(-beta * E[ii])
    Pe = Pe / np.sum(Pe)
    return Pe


def probablistic_simulation(J):
    Pr = np.zeros(2**NM)
    m = np.ones(J.shape[0]) 
    h = np.zeros((J.shape[0], 1))
    Look = [2**i for i in range(J.shape[0]-1, -1, -1)] # bin identifier vector

    for ii in range(0, NT): # samples loop
        for jj in range(0, J.shape[0]): # update all the spins one by one sequentially

            # synapse equation for jj-th pbit
            I = np.dot(J[jj, :], m) + h[jj]
            # neuron equation for jj-th pbit
            m[jj] = np.sign(np.tanh(beta * I) - 2 * random.random() + 1)

        # when we have updated all the spins at once we get one sample
        # given a state of the spins find out the bin number in the 2^NM space
        kk = 1 + np.dot(Look, (1 + m) / 2)

        # increment the corresponding bin
        Pr[int(kk)-1] += 1

    # Normalize probabilities
    Pr = Pr / np.sum(Pr)
    return Pr


def restructure_probs(Pr, Pe=None):
    #only get the last 3 spins, so 8 values.
    Pr3 = np.zeros(8)
    Pe3 = np.zeros(8)

    # normalise over A, B and C
    for i in range(len(Pr)):
        binary_str = np.binary_repr(i, width=NM)
        handle_bit = binary_str[4]
        new_binary_str = binary_str[-3:] # last 3 bits (ABC)
        new_index = int(new_binary_str, 2) # convert binary to int

        if handle_bit == '1': # only add if handle bit is 1
            Pr3[new_index] += Pr[i]
            if Pe is not None:
                Pe3[new_index] += Pe[i]

    # re-normalise probabilities
    Pr = Pr3 / np.sum(Pr3)
    if Pe is not None:
        Pe = Pe3 / np.sum(Pe3)
        return Pr, Pe
    else:
        return Pr

def restructure_probs_no_handle(Pr, Pe=None):
    #only get the last 3 spins, so 8 values.
    Pr4 = np.zeros(16)
    Pe4 = np.zeros(16)

    # normalise over A, B and C
    for i in range(len(Pr)):
        binary_str = np.binary_repr(i, width=NM)
        new_binary_str = binary_str[-4:] # last 3 bits (ABC)
        new_index = int(new_binary_str, 2) # convert binary to int
        Pr4[new_index] += Pr[i]
        if Pe is not None:
            Pe4[new_index] += Pe[i]

    # re-normalise probabilities
    Pr = Pr4 / np.sum(Pr4)
    if Pe is not None:
        Pe = Pe4 / np.sum(Pe4)
        return Pr, Pe
    else:
        return Pr


def plot_probs(Pr, Pe=None):
    x_positions = np.arange(len(Pr))
    width = 0.4  # Adjust the width as needed
    labels = ["000", "001", "010", "011", "100", "101", "110", "111"]

    plt.figure()

    plt.bar(x_positions + width/2, Pr, width, label='Probabilistic')
    if Pe is not None: 
        plt.bar(x_positions - width/2, Pe, width, label='Exact')

    plt.xticks(x_positions, labels)
    plt.xlabel('States')
    plt.ylabel('Probabilities')
    plt.legend()
    plt.title('Probabilistic Emulation of a 3-Spin System')
    plt.show()


def plot_probs_no_handle(Pr, Pe=None):
    x_positions = np.arange(len(Pr))
    width = 0.4  # Adjust the width as needed
    labels = ["0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111","1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111"]

    plt.figure()

    plt.bar(x_positions + width/2, Pr, width, label='Probabilistic')
    if Pe is not None: 
        plt.bar(x_positions - width/2, Pe, width, label='Exact')

    plt.xticks(x_positions, labels)
    plt.xlabel('States')
    plt.ylabel('Probabilities')
    plt.legend()
    plt.title('Probabilistic Emulation of a 3-Spin System')
    plt.show()


def sample_spins(Pr, n):
    samples = []
    for i in range(n):
        value = np.random.choice(range(0, len(Pr)), size=1, p=Pr)
        output = np.binary_repr(value[0])
        
        while len(output) < 3:
            output = "0" + output
        samples.append(output)

    return samples


def W_from_J(J, Wx, Wy):
    Jx = int(J.shape[0]/2)
    W = J[Jx:Wx+Jx,:4]
    return W


def J_from_W(W, Jx, Jy):
    J = np.zeros([Jx, Jy])
    for i in range(Jx):
        for j in range(Jy):
            if i >= Jx/2 and j < Jy/2:
                # print("if: ", i, j)
                W_i = int(i - Jx/2)
                J[i][j] = W[W_i][j]
            elif i < Jx/2 and j >= Jy/2:
                W_j = int(j - Jy/2)
                # print("else:, ", i, j)
                J[i][j] = W.T[i][W_j]
    return J


def main(tt_values, nr_samples):
    truth_table = np.array([[1,-1,-1,tt_values[0]],
                            [1,-1,1,tt_values[1]],
                            [1,1,-1,tt_values[2]],
                            [1,1,1,tt_values[3]]])

    W = [[-1, 1, -0.3, -1],
        [-0.5, -1, -1, 1,],
        [-1, -0.6, 1, 1],
        [-1, 1, -1, 1]]
    W = np.array(W)

    # J = calculate_J(truth_table, 4, 4)
    # print(J)

    # W = W_from_J(J, 4, 4)
    # print(W)

    J = J_from_W(W, 8, 8)
    print(J)

    Pe = exact_simulation(J)

    Pr = probablistic_simulation(J)
    # Pr, Pe = restructure_probs_no_handle(Pr, Pe)
    # plot_probs_no_handle(Pr, Pe)

    Pr, Pe = restructure_probs(Pr, Pe)
    plot_probs(Pr, Pe)

    samples = sample_spins(Pr, nr_samples)
    return samples


if __name__ == '__main__':
    tt_values = sys.argv[2:]
    tt_values = [int(x) for x in tt_values]
    nr_samples = int(sys.argv[1])
    samples = main(tt_values, nr_samples)
    for sample in samples:
        sys.stdout.write(sample+" ")

    #TODO: add argument to decide how many samples you want. then return a list of samples.
