import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from logistic import logistic_model

result = logistic_model(X,y)
print(result)