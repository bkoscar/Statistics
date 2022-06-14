import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import joblib
import warnings
warnings.filterwarnings('ignore')


#Leer el conjunto de datos
data = pd.read_csv('advertising.csv')

# Mostrar los primeros 5 valores del dataset
# print(data.head(5))
# Graficos de puntos respectivamente a cada columna vs la columna Sales
# plt.scatter(data['TV'].values, data['Sales'].values, label = 'TV')
# plt.scatter(data['Radio'].values, data['Sales'].values,label = 'Radio')
# plt.scatter(data['Newspaper'].values, data['Sales'].values, label = 'Newspaper')
# plt.legend()
# plt.show()

# Estatistica descriptiva
# print('Descriptive Statistics')
# print(data.describe())
# corr_test = pearsonr( x = data['Newspaper'], y = data['Sales'])
# print(f'Pearson correlation: {corr_test[0]}')
# print(f'P-value: {corr_test[1]}')

# Del analisis estadistico anterior se obtuvo que la variable TV es la que presenta mayor correlacion con la variable dependiente Sales, se realziara un analisis de regresion lineal simple
# Particion del dataset en datos de entrenamiento y de test
X = data[['TV']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(
   X,
   y,
   test_size= 0.2,
   random_state=42,
   shuffle=True 
)
# Ajuste del modelo
model = LinearRegression()
model.fit(X_train,y_train)

# Coeficientes del modelo lineal
# print(f'Coeficiente 1: {model.coef_}, Coeficiente 0 {model.intercept_} ')
# print(f'Coeficiente de determinacion de prediccion {model.score(X_train,y_train)}')

# Guardar el modelo para posteriormente utilizarlo
joblib.dump(model,"regression.pkl")