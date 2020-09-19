import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy as sp

class struct():
    pass

def getComplexNums(B=4):
    """
    This function generates the complex numbers of the lattice representing the constellation
    map.
    :param B: 2^B-QAM for PAPR calculation purposes.
    :return: The complex numbers in the lattice.
    """
    complex_numbers = []

    for k in np.linspace(-2**(int(B/2)) + 1, 2**(int(B/2)) - 1, 2**(int(B/2))).astype(int):
        for l in np.linspace(-2**(int(B/2)) + 1, 2**(int(B/2)) - 1, 2**(int(B/2))).astype(int):
            complex_numbers = complex_numbers + [[k, l]]

    return complex_numbers

def getBasicMap(n):
    """
    This function generates one of the 8 basic 2x2 gray coded maps.
    :param n: The choice of the basic 2x2 map.
    :return: The basic 2x2 map
    """
    R = [struct() for i in range(8)]

    R[0].mat = [[[0, 0], [0, 1]], [[1, 0], [1, 1]]]
    R[1].mat = [[[0, 0], [1, 0]], [[0, 1], [1, 1]]]
    R[2].mat = [[[1, 1], [0, 1]], [[1, 0], [0, 0]]]
    R[3].mat = [[[1, 1], [1, 0]], [[0, 1], [0, 0]]]
    R[4].mat = [[[0, 1], [0, 0]], [[1, 1], [1, 0]]]
    R[5].mat = [[[1, 0], [0, 0]], [[1, 1], [0, 1]]]
    R[6].mat = [[[0, 1], [1, 1]], [[0, 0], [1, 0]]]
    R[7].mat = [[[1, 0], [1, 1]], [[0, 0], [0, 1]]]

    return R[n].mat

def getCombinations(B=4):
    """
    This function generates all possible combinations of bit positions to be kept constant
    while incrementally generating the constellation map.
    :param B: 2^B-QAM for PAPR calculation purposes.
    :return: All possible combinations.
    """
    permutations = list(itertools.permutations(np.linspace(0, B - 1, B).astype(int)))
    combinations = []
    for j in range(len(permutations)):
        tmp = []
        for k in range(int(B / 2)):
            tmp = tmp + [set([permutations[j][2 * k], permutations[j][2 * k + 1]])]
        combinations = combinations + [tmp]

    uniq_combinations = [combinations[0]]
    for j in range(len(combinations)):
        flag = 0
        for k in range(len(uniq_combinations)):
            if (np.sum([combinations[j][t] == uniq_combinations[k][t] for t in range(int(B / 2))]) == int(B / 2)):
                flag = 1
                break
        if (flag == 0):
            uniq_combinations = uniq_combinations + [combinations[j]]

    return uniq_combinations

def getMap(B=4, basic_map_num=[1, 2], combination=[{0, 1}, {2, 3}]):
    """
    This function generates a constellation map given the choice of basic maps to incrementally
    generate the map and the choice of bits to be kept constant in any incremental step.
    :param B: 2^B-QAM for PAPR calculation purposes.
    :param basic_map_num: The choice of basic maps to incrementally
    generate the map
    :param combination: The choice of bits to be kept constant in any incremental step
    :return: The constellation map, the complex numbers part and the bits part.
    """
    Rfinal = [[list(np.zeros(B)) for j in range(2**(int(B/2)))] for i in range(2**(int(B/2)))]

    Rtmp = getBasicMap(basic_map_num[0])
    for k in range(int(B/2)-1):
        Rtoadd = getBasicMap(basic_map_num[k+1])
        Rtmp_new = [[[] for j in range((2**(k+1))*2)] for i in range((2**(k+1))*2)]

        zi = 0
        zj = 0
        for i in range((2**(k))*2):
            for j in range((2**(k))*2):
                Rtmp_new[i+zi][j+zj] = Rtoadd[int(zi/len(Rtmp))][int(zj/len(Rtmp))] + Rtmp[i][j]

        zi = len(Rtmp)
        zj = 0
        for i in range((2**(k))*2):
            for j in range((2**(k))*2):
                Rtmp_new[i+zi][j+zj] = Rtoadd[int(zi/len(Rtmp))][int(zj/len(Rtmp))] + Rtmp[len(Rtmp)-1-i][j]

        zi = 0
        zj = len(Rtmp)
        for i in range((2**(k))*2):
            for j in range((2**(k))*2):
                Rtmp_new[i+zi][j+zj] = Rtoadd[int(zi/len(Rtmp))][int(zj/len(Rtmp))] + Rtmp[i][len(Rtmp)-1-j]

        zi = len(Rtmp)
        zj = len(Rtmp)
        for i in range((2**(k))*2):
            for j in range((2**(k))*2):
                Rtmp_new[i+zi][j+zj] = Rtoadd[int(zi/len(Rtmp))][int(zj/len(Rtmp))] + Rtmp[len(Rtmp)-1-i][len(Rtmp)-1-j]

        Rtmp = Rtmp_new

    for i in range(len(Rtmp)):
        for j in range(len(Rtmp)):
            for l in range(len(combination)):
                Rfinal[i][j][np.sort(list(combination[l]))[0]] = Rtmp[i][j][2*l]
                Rfinal[i][j][np.sort(list(combination[l]))[1]] = Rtmp[i][j][2*l+1]


    A = getComplexNums(B)
    complex_nums = []
    corresposing_bits = []
    L = int(np.sqrt(len(A)))
    for k in range(len(A)):
        complex_nums = complex_nums + [complex(A[k][0], A[k][1])]
        corresposing_bits = corresposing_bits + [Rfinal[np.mod(k, L)][int((k-np.mod(k, L))/L)]]

    return Rfinal, complex_nums, corresposing_bits

def plot(R, B):
    """
    This function plots the Gray coded constellation map.
    :param R: A list of bit strings for every appropriate lattice point on the complex plane.
    :param B: 2^B-QAM for PAPR calculation purposes.
    :return: A plot of the constellation map with Gray Coding.
    """
    complex_nums = getComplexNums(B)
    plt.figure()
    Z = np.real(complex_nums)
    plt.scatter(Z[:,0], Z[:,1])
    L = int(np.sqrt(len(complex_nums)))
    for i in range(L):
        for j in range(L):
            plt.text(Z[L*i+j,0],Z[L*i+j,1],''.join([(str(t)) for t in R[i][j]]))


def getaBasicMapCombinations(B):
    """
    This function gets all possible combinations of the basic 2x2 constellation.
    :param B: 2^B-QAM for PAPR calculation purposes.
    :return: All possible combinations.
    """
    basicMapCombinations = []
    if(B==4):
        for i in range(8):
            for j in range(8):
                basicMapCombinations = basicMapCombinations + [[i,j]]
    elif(B==6):
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    basicMapCombinations = basicMapCombinations + [[i,j,k]]

    return basicMapCombinations

def papr_calc(message, B, complex_nums, corresposing_bits):
    """
    This function calculates the PAPR given the message and constellation map.
    :param message: A list of 0's and 1's.
    :param B: 2^B-QAM for PAPR calculation purposes.
    :param complex_nums: The constellation map (complex numbers part).
    :param corresposing_bits: The constellation map (bits part).
    :return: The PAPR number.
    """
    complex_message = [complex_nums[corresposing_bits.index(list(message[B*k:B*(k+1)]))] for k in range(int(len(message)/B))]
    peak_power = np.max(np.abs(sp.ifft(complex_message)))**2
    avg_power = np.linalg.norm(complex_message)**2/len(complex_message)
    papr = peak_power/avg_power

    return papr

def getOptMap(message, B):
    """
    This function calculates the optimal constellation map for a given message under
    2^B-QAM OFDM scheme.
    :param message: A list of 0's and 1's.
    :param B: 2^B-QAM for PAPR calculation purposes.
    :return: The optimal constellation map (with Gray Coding), and a plot of it.
    """
    if(np.mod(len(message),B)!=0):
        print('The length of the message must be an integer multiple of B!')
        return -1

    if(np.mod(B,2)!=0):
        print('B must be even!')
        return -1

    if (B>6):
        print('B can be at most 6, for computational purposes.')
        return -1

    basic_map_nums = getaBasicMapCombinations(B)
    combinations = getCombinations(B)
    papr_min = np.inf
    Ropt  = []
    complex_nums_opt = []
    corresposing_bits_opt = []

    for basic_map_num in basic_map_nums:
        for combi in combinations:
            Rfinal, complex_nums, corresposing_bits = getMap(B, basic_map_num, combi)
            papr = papr_calc(message, B, complex_nums, corresposing_bits)
            if(papr<papr_min):
                papr_min = papr
                Ropt = Rfinal
                complex_nums_opt = complex_nums
                corresposing_bits_opt = corresposing_bits

    plot(Ropt, B)
    print('The papr value is ' + str(papr_min))
    return Ropt, complex_nums_opt, corresposing_bits_opt

if(__name__=='__main__'):
    B = 4
    message = np.random.randint(0, 2, B * 100)
    getOptMap(message, B)
