import pandas as pd  
import matplotlib.pyplot as plt 
import joblib
import warnings
warnings.filterwarnings('ignore')

# Cargar el dataset
data = pd.read_csv('advertising.csv')
# Extraccion de la variable TV y Sales
tv_data = data.iloc[:,[0,3]]

# Se carga el modelo previamente entrenado
model = joblib.load('regression.pkl')

# Se pasan los valores de la variable TV al modelo y que este realice una prediccion 
s = model.predict(tv_data['TV'].values.reshape(-1,1))
# Se grafica el modelo y el dataset
plt.figure(dpi = 80)
plt.scatter(tv_data['TV'].values,tv_data['Sales'].values,color ='c' ,label = 'Values of TV')
plt.plot(tv_data['TV'].values, s,color = 'k', label = 'Regression')
plt.legend()
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()