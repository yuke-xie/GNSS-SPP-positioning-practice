# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  $filename :                                                                 +
#                                                                              +
#            Copyright (c) 2021.  by Y.Xie, All rights reserved.               +
#                                                                              +
#   references :                                                               +
#       [1] GPX The GNSS Time transformation https://www.gps.gov/technical/icwg/
#                                                                              +
#   version : $Revision:$ $Date:$                                              +
#   history : 2020/12/11  1.0  new                                             +
#             26/01/2021, 16:57  1.1  modify                               +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import georinex as gr
import time
import pickle
import numpy as np
from gnsstimetrans import utctoweekseconds
from gpspos import gpspos_ecef, correctPosition


def SellectSystem(all_sat, system):
    system_letter = {
        'GPS': "G",
        'GLONASS': "R",
        'Beidou': "C",
        'Galileo': "E",
    }
    letter = system_letter.get(system, None)

    all_sat = np.array(all_sat)
    want_sat = []
    for i in range(len(all_sat)):
        if all_sat[i][0] == letter:
            want_sat = np.append(want_sat, all_sat[i])
    return want_sat


if __name__ == '__main__':
    # load observation
    # obs = gr.load('data/INSA002mA.21o')
    # pickle.dump(obs, open('data/INSA002mA_21o_temp.txt', 'wb'))
    obs = pickle.load(open('data/INSA003mA_21o_temp.txt', 'rb'))
    # print('finished')
    # end = time.clock()
    # print(end - start)

    # load correspond ephemeris
    epoch_first = str(np.array(obs.time[0]))[0:19]
    print(epoch_first)
    epoch_last = str(np.array(obs.time[-1]))[0:19]
    print(epoch_last)
    eph = gr.load('data/INSA003mA.21m', tlim=[epoch_first, epoch_last])

    # list all available satellites
    GPS = SellectSystem(eph.sv, 'GPS')
    print('\nAvailable satellite:\n', GPS)
    P_obs = np.zeros([len(GPS), 1])
    P_computed = np.zeros([len(GPS), 1])
    A = np.zeros([len(GPS), 4])
    delta_P = np.zeros([len(GPS), 1])
    A[:, 3] = 299792458
    sat_pos = np.zeros([len(GPS), 3])
    for n in range(len(GPS)):
        # print('Use satellite:', GPS[n])
        ## calculate the position of satellite GPS_n
        GPS_n = eph.sel(sv=GPS[n]).dropna(dim='time', how='all')
        soW = utctoweekseconds(epoch_first, 0)[1]
        # print('Time of the week:', soW)
        GPS_n_pos_raw = gpspos_ecef(GPS_n, soW)
        # print('\nPosition of satellite', GPS[n], ':\n', GPS_n_pos)

        ## get the pseudo-range of satellite GPS_n
        pseudo_range = np.array(obs['C1C'].sel(sv=GPS[n]))
        # print('\nnumber of satellite: ', pseudo_range.shape[1])
        # print('\nnumber of epoch: ', pseudo_range.shape[0])
        pseudo_GPS_n = pseudo_range[0]
        time_of_flight = pseudo_GPS_n / 299792458
        GPS_n_pos = correctPosition(GPS_n_pos_raw, time_of_flight)
        # print('\nPseudo-range of satellite GPS_n:\n', pseudo_GPS_n)

        ## test the range based on receiver's header position
        pos_header = obs.position
        # print(pos_header)
        GPS_n_range_header = np.linalg.norm(GPS_n_pos-pos_header)
        # print('\nRange based on header posion:\n', GPS_n_range_header)

        # evaluate the difference between pseudo-range

        diff = pseudo_GPS_n - GPS_n_range_header
        print(diff)
        # print('\nDifferance between pseudo-range', GPS[n], ':\n', diff, '[m]')

        A[n, 0:3] = (pos_header - GPS_n_pos) / pseudo_GPS_n
        P_obs[n] = pseudo_GPS_n
        P_computed[n] = GPS_n_range_header
        sat_pos[n, :] = GPS_n_pos


    # apply least square adjustment
    max_iter = 5
    x = obs.position
    for i in range(max_iter):
        delta_P = P_obs - P_computed
        delta_x = np.linalg.inv(A.T@A)@A.T@delta_P
        x = x + delta_x[0:3].T
        receiver_pos = np.tile(x, (len(GPS), 1))
        A = np.hstack(((receiver_pos - sat_pos) / np.tile(P_obs, (1, 3)), np.tile([299792458], (len(GPS), 1))))
        P_computed = np.linalg.norm(sat_pos-receiver_pos, axis=1).reshape(len(GPS), 1)
    print('delta X after ', i, ' times iteration: ', delta_x)
    print('result position', x)
    print('true position  ', obs.position)



