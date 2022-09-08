from Diabetes_Experiment_PWB_Test import ex
import json
import sys

def loadConfig(exNum):
    with open('../%d'%exNum + '/config.json') as json_file:
        return json.load(json_file)
        
def main():
    r = ex.run(config_updates=loadConfig(21))


main()
