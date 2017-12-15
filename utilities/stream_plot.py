from ggplot import meat
import pandas as pd
from ggplot import *
import matplotlib.pyplot as plt

meat_lng = pd.melt(meat, id_vars=['date'])
print(meat_lng)
g = ggplot(aes(x='date', y='value', colour='variable'), data=meat_lng) + geom_line()

print(g)