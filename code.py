# quero fazer uma regressão linear em Python, em logs, de Y em função de X1, X2, X3 e X4
# criadas aleatoriamente e que respeitem aos pressupostos clássicos de homocedasticidade e não correlação serial e normalidade dos resíduos.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
n = 1000
X1 = np.random.rand(n) * 100
X2 = np.random.rand(n) * 50
X3 = np.random.rand(n) * 20
X4 = np.random.rand(n) * 10
# Coefficients
beta0 = 2
beta1 = 0.5
beta2 = 0.3
beta3 = 0.2
beta4 = 0.1
# Generate Y with some noise
epsilon = np.random.normal(0, 0.5, n)
Y = np.exp(beta0 + beta1 * np.log(X1) + beta2 * np.log(X2) + beta3 * np.log(X3) + beta4 * np.log(X4) + epsilon)
data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4})
data['log_Y'] = np.log(data['Y'])
data['log_X1'] = np.log(data['X1'])
data['log_X2'] = np.log(data['X2'])
data['log_X3'] = np.log(data['X3'])
data['log_X4'] = np.log(data['X4'])
X = data[['log_X1', 'log_X2', 'log_X3', 'log_X4']]
X = sm.add_constant(X)
y = data['log_Y']
model = sm.OLS(y, X).fit()
print(model.summary())
# Residuals
residuals = model.resid
fitted = model
fitted_values = model.fittedvalues
# Plot residuals vs fitted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=fitted_values, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()
# Histogram of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Histogram of Residuals')
plt.show()
# Q-Q plot
plt.figure(figsize=(10, 6))
qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()
# Breusch-Pagan test for homoscedasticity
bp_test = het_breuschpagan(residuals, X)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
print(dict(zip(labels, bp_test)))
# Durbin-Watson test for autocorrelation
dw_stat = durbin_watson(residuals)
print(f'Durbin-Watson statistic: {dw_stat}')
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_train = sm.OLS(y_train, X_train).fit()
y_pred = model_train.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# Plot predicted vs actual
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.show()
# Save the model summary to a text file
with open('model_summary.txt', 'w') as f:
    f.write(model.summary().as_text())
# Save the data to a CSV file
data.to_csv('regression_data.csv', index=False)
# Save the plots
plt.figure(figsize=(10, 6))
sns.scatterplot(x=fitted_values, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.savefig('residuals_vs_fitted.png')
plt.close()
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Histogram of Residuals')
plt.savefig('histogram_of_residuals.png')
plt.close()
plt.figure(figsize=(10, 6))
qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.savefig('qq_plot_of_residuals.png')
plt.close()
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.savefig('predicted_vs_actual.png')
plt.close()
print("All outputs have been saved to files.")
# Save the model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model has been saved to 'linear_regression_model.pkl'.")
# Load the model (example)
with open('linear_regression_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print(loaded_model.summary())
print("Model has been loaded from 'linear_regression_model.pkl'.")
print("Model has been saved to 'linear_regression_model.pkl'.")
