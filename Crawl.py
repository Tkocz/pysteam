from __future__ import print_function
from steamwebapi.api import ISteamUser, IPlayerService, ISteamUserStats
import sys
import time

import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark.sql import SparkSession

steamID = ['76561198048730871', '76561198180821795', '76561198008911412']
steamuserinfo = ISteamUser()
playerserviceinfo = IPlayerService()

try:
    y = 0
    for n in steamID:

        #usersummary = steamuserinfo.get_player_summaries(n)['response']['players']
        #owned_games = playerserviceinfo.get_owned_games(n)['response']['games']
        friendslist = steamuserinfo.get_friends_list(n)['friendslist']['friends']
        steamid_list =[]
        game_list = []
        #for i in owned_games:
        #    game_list.append(i['name'])
        for i in friendslist:
            if i['steamid'] not in steamID:
                steamid_list.append(i['steamid'])
                steamID.append(i['steamid'])
        print("\n".join(game_list))
        print("\n".join(steamid_list))
        #print(usersummary)
        #print(owned_games)
        print(friendslist)
        if y > 5:
            break
        y += 1

finally:
        print('-----------------------------')
        print(len(steamID))
        print(steamID)



# LAMBDA = 0.01   # regularization
# np.random.seed(42)
#
#
# def rmse(R, ms, us):
#     diff = R - ms * us.T
#     return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))
#
#
# def update(i, vec, mat, ratings):
#     uu = mat.shape[0]
#     ff = mat.shape[1]
#
#     XtX = mat.T * mat
#     Xty = mat.T * ratings[i, :].T
#
#     for j in range(ff):
#         XtX[j, j] += LAMBDA * uu
#
#     return np.linalg.solve(XtX, Xty)
#
#
# if __name__ == "__main__":
#
#     """
#     Usage: als [M] [U] [F] [iterations] [partitions]"
#     """
#
#     print("""WARN: This is a naive implementation of ALS and is given as an
#       example. Please use pyspark.ml.recommendation.ALS for more
#       conventional use.""", file=sys.stderr)
#
#     spark = SparkSession\
#         .builder\
#         .appName("PythonALS")\
#         .getOrCreate()
#
#     sc = spark.sparkContext
#
#     M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
#     U = int(sys.argv[2]) if len(sys.argv) > 2 else 500
#     F = int(sys.argv[3]) if len(sys.argv) > 3 else 10
#     ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
#     partitions = int(sys.argv[5]) if len(sys.argv) > 5 else 2
#
#     print("Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %
#           (M, U, F, ITERATIONS, partitions))
#
#     R = matrix(rand(M, F)) * matrix(rand(U, F).T)
#     ms = matrix(rand(M, F))
#     us = matrix(rand(U, F))
#
#     Rb = sc.broadcast(R)
#     msb = sc.broadcast(ms)
#     usb = sc.broadcast(us)
#
#     for i in range(ITERATIONS):
#         ms = sc.parallelize(range(M), partitions) \
#                .map(lambda x: update(x, msb.value[x, :], usb.value, Rb.value)) \
#                .collect()
#         # collect() returns a list, so array ends up being
#         # a 3-d array, we take the first 2 dims for the matrix
#         ms = matrix(np.array(ms)[:, :, 0])
#         msb = sc.broadcast(ms)
#
#         us = sc.parallelize(range(U), partitions) \
#                .map(lambda x: update(x, usb.value[x, :], msb.value, Rb.value.T)) \
#                .collect()
#         us = matrix(np.array(us)[:, :, 0])
#         usb = sc.broadcast(us)
#
#         error = rmse(R, ms, us)
#         print("Iteration %d:" % i)
#         print("\nRMSE: %5.4f\n" % error)
#
#     spark.stop()