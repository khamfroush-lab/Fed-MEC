import os
import pandas as pd
import glob

def get_data(algorithm_code: str="main") -> None:
    assert algorithm_code in ["main", "dcfl"]
    
    ## Establish the dirs we're interested in and the rule we're abiding by.
    filenames = []
    dirs = ["Cifar-0.2-0.4", "Cifar-0.6", "Cifar-0.8", "Cifar-1"]
    rule = lambda string: ("Global" in string)    and \
                          ("Final" not in string) and \
                          (algorithm_code in string.lower())

    ## Grab the files that meet the criteria and store their paths.
    for d in dirs:
        paths = glob.glob(os.path.join(d, "*.json"))
        for path in paths:
            f = path.split(os.path.sep)[-1]
            if rule(f):
                filenames.append(path)
                
    ## Read the JSON files using Pandas and concatenate them.
    data = pd.DataFrame()
    for json in filenames:
        json_df = pd.read_json(json)
        data = pd.concat([data, json_df])
        
    ## Save the merged_data.
    data = data.reset_index(drop=True)
    print(data.head())
    data.to_csv(f"cifar-{algorithm_code}-data.csv")

def get_final_data(algorithm_code: str="main") -> None:
    assert algorithm_code in ["main", "dcfl", "final"]
    
    ## Establish the dirs we're interested in and the rule we're abiding by.
    filenames = []
    dirs = ["Cifar-0.2-0.4", "Cifar-0.6", "Cifar-0.8", "Cifar-1"]
    rule = lambda string: ("Global" in string) and \
                          ("Final"  in string) and \
                          (algorithm_code in string.lower())

    ## Grab the files that meet the criteria and store their paths.
    for d in dirs:
        paths = glob.glob(os.path.join(d, "*.json"))
        for path in paths:
            f = path.split(os.path.sep)[-1]
            if rule(f):
                filenames.append(path)
    
    ## Read the JSON files using Pandas and concatenate them.
    data = pd.DataFrame()
    for json in filenames:
        json_df = pd.read_json(json)
        data = pd.concat([data, json_df])

    ## Save the merged_data.
    data = data.reset_index(drop=True)
    print(data.head())
    data.to_csv(f"cifar-{algorithm_code}-final-data.csv")

if __name__ == "__main__":
    get_data("main")
    get_data("dcfl")
    get_final_data("main")
    get_final_data("dcfl")