
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/Ecommerce Customers.csv")

df.info()

df.describe()

"""#EDA

"""

sns.jointplot(x ="Time on App" , y = "Yearly Amount Spent", data = df, alpha = 0.5 )

sns.pairplot(df, kind ='scatter', plot_kws = {'alpha': 0.4})

sns.lmplot(x='Length of Membership',
           y = 'Yearly Amount Spent',
           data = df ,
           scatter_kws ={'alpha':0.3})

from sklearn.model_selection import train_test_split

X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state= 42)

"""#Training the model

"""

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

lm.coef_

edf = pd.DataFrame(lm.coef_,X.columns, columns=['coef'])
print(edf)

# predications

predications = lm.predict(X_test)

predications

sns.scatterplot(x = predications , y = y_test)
plt.xlabel("predictions")

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

print("Mean Absolute Error:", mean_absolute_error(y_test, predications))
print("Mean Squared Error:", mean_squared_error(y_test,predications))
print("RMSE Absolute Error:", math.sqrt(mean_squared_error(y_test,predications)))

# residuals

residuals = y_test - predications

residuals

sns.distplot(residuals, bins = 20)

import pylab
import scipy.stats as stats

stats.probplot(residuals, dist='norm', plot=pylab)
pylab.show()

