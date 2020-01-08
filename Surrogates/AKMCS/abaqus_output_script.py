# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import os
import numpy as np


def ODBAnalysis():
    import visualization

    # In the input script, the jobname is defined as 'fire_analysis'. So, open the odb file with the same jobname.
    odbname = 'fire_analysis.odb'
    o1 = session.openOdb(name=odbname)

    odb = session.odbs[odbname]

    # Extract the temperature and y-component of displacement from history output at midpoint of beam
    xy0 = session.XYDataFromHistory(name='XYData-1', odb=odb,
                                    outputVariableName='Nodal temperature: NT11 PI: BEAM-1 Node 2 in NSET MIDPOINT',
                                    steps=('ThermalStep',), )
    xy1 = session.XYDataFromHistory(name='XYData-2', odb=odb,
                                    outputVariableName='Spatial displacement: U2 PI: BEAM-1 Node 2 in NSET MIDPOINT',
                                    steps=('ThermalStep',), )

    session.odbs[odbname].close()

    temps = xy0.data
    disps = xy1.data

    outfilename = 'time_temp_disp_data.csv'
    outfile = open(outfilename, 'w')

    # for i in range(len(disps)):
    #     outfile.write('%10.6f, %10.6f, %10.6f \n' % (disps[i][0] - 1, temps[i][1], disps[i][1] - disps[0][1]))

    # Compute the maximum allowable displacement
    length_of_beam = 1  # in m
    depth_of_beam = 0.035  # in m
    max_allowable_disp = length_of_beam ** 2 / (400 * depth_of_beam)
    disps_array = np.array(disps)

    y_akmcs = -max(map(abs, (disps_array[:, 1] - disps_array[0, 1]))) + max_allowable_disp
    # if disps_array[-1, 0] < 1:
    #     y_akmcs = -np.random.uniform(0, 0.1)*max_allowable_disp

    outfile.write('%10.6f \n' % y_akmcs)
    outfile.close()


if __name__ == "__main__":
    ODBAnalysis()
