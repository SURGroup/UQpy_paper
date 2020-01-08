import fire
import os
import numpy as np
import subprocess


def extract_disp_temp_output(index):
    # index = int(index)
    output_script_path = os.path.join(os.getcwd(), 'abaqus_output_script.py')
    command = "abaqus cae nogui=" + output_script_path
    try:
        out = os.system(command)
        #print('Out: ', out)
        if out == 0:
            #print('Example: Successful output extraction.')
            outfilename = 'time_temp_disp_data.csv'
            data = np.genfromtxt(outfilename, delimiter=',')
            return data
        # subprocess.run(command, shell=True)
    except OSError as err:
        print(err)
        return np.array([100, 100, 10000])


if __name__ == '__main__':
    fire.Fire(extract_disp_temp_output)
