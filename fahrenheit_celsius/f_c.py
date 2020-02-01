import numpy as np
import pandas as pd
from os import getcwd, listdir

data_path = getcwd() + "/fahrenheit_celsius"

celsius = [i for i in range(-50, 51)]

fahrenheit = [1.8*i + 32 for i in celsius]

df = pd.DataFrame.from_dict({'celsius': celsius, 'fahrenheit': fahrenheit})

df.to_csv(f"{data_path}/temperatures.csv", index=False)
