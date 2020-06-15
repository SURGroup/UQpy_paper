import fire
import os


def run_fire_analysis(index):
    index = int(index)
    print('Example: Started analysis for sample %d' % index)
    abaqus_script_path = os.path.join(os.getcwd(), 'InputFiles', 'abaqus_input_subset_sfe_' + str(index) + ".py")
    command = "abaqus cae nogui=" + abaqus_script_path
    # command = "abaqus cae script=" + abaqus_script_path
    try:
        o = os.system(command)
        if o == 0:
            print('Example: Ran successfully.')
    except Exception as err:
        print(err)


if __name__ == '__main__':
    fire.Fire(run_fire_analysis)
