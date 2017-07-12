from Ranking import *
import pandas as pd

rank = Rank()
predictions = pd.DataFrame([[1, 1, 1.0, 1.0], [1, 2, 0.0, 0.5], [1, 3, 0.0, 0.1], [2, 2, 1.0, 0.5],
                           [2, 3, 0.0, 0.1], [3, 1, 0.0, 1.0], [3, 2, 0.0, 0.5], [3, 3, 1.0, 0.1]], columns=['steamid', 'appid', 'rating', 'prediction'])
users = predictions.where(predictions.rating == 1.0)

predictions = predictions.append(pd.DataFrame([[2, 1, 1.0, 1.0]], columns=['steamid', 'appid', 'rating', 'prediction'])).sort_values(['steamid', 'prediction'], ascending=[True, False])
del users['rating']
del users['prediction']
print(predictions)
dict = {'test': predictions}
result = rank.rank(dict, users, 0)

print(result)