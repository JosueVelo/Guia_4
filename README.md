### **Guía N.º 04: Análisis Exploratorio de Datos**

**Integrante:**
- Josué Nehemías Velo Poma.

### **Variaciones Mensuales en el Número de Turistas en los 7 Estados Más Visitados de EE.UU.**

### **1. Generación o captura**

- En 2017, los datos climáticos diarios en EE. UU. fueron capturados por estaciones meteorológicas terrestres distribuidas por todo el país. Estas estaciones midieron variables clave como la temperatura máxima y mínima, la precipitación diaria total, la cantidad de nieve y la profundidad de la nieve. La generación o captura de estos datos fue crucial, ya que los fenómenos climáticos pueden variar ampliamente según la ubicación y el día, y la medición precisa y oportuna de estas variables permite documentar las condiciones meteorológicas diarias con exactitud.

### **2. Recolección de datos**

- Después de la captura de los datos en las estaciones meteorológicas, la información de 2017 fue recopilada y centralizada por el GHCNd. Este proceso implica la recolección diaria de datos de múltiples estaciones en diferentes ubicaciones, unificándolos bajo un esquema común para que puedan ser comparables y coherentes. Durante esta fase, los datos fueron sometidos a revisiones de calidad para detectar posibles errores en las mediciones o en la transmisión de los datos, asegurando que las estadísticas diarias de 2017 estuvieran completas y precisas.

### **3. Limpieza de Datos**

#### **3.1 Nivel I: Revisión Inicial de Datos**


```python
!pip install category_encoders
```

    Collecting category_encoders
      Downloading category_encoders-2.6.3-py2.py3-none-any.whl.metadata (8.0 kB)
    Requirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (1.26.4)
    Requirement already satisfied: scikit-learn>=0.20.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (1.3.2)
    Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (1.13.1)
    Requirement already satisfied: statsmodels>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (0.14.3)
    Requirement already satisfied: pandas>=1.0.5 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (2.1.4)
    Requirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (0.5.6)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.5->category_encoders) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.5->category_encoders) (2024.2)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.5->category_encoders) (2024.1)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from patsy>=0.5.1->category_encoders) (1.16.0)
    Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.20.0->category_encoders) (1.4.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.20.0->category_encoders) (3.5.0)
    Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.10/dist-packages (from statsmodels>=0.9.0->category_encoders) (24.1)
    Downloading category_encoders-2.6.3-py2.py3-none-any.whl (81 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m81.9/81.9 kB[0m [31m3.6 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: category_encoders
    Successfully installed category_encoders-2.6.3



```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBRegressor
from category_encoders import LeaveOneOutEncoder
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
```


```python
data = pd.read_csv('weather.csv')
data.head()
```





  <div id="df-e64bc980-059d-418e-86d1-0b90431328f0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>station</th>
      <th>state</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>elevation</th>
      <th>date</th>
      <th>TMIN</th>
      <th>TMAX</th>
      <th>TAVG</th>
      <th>AWND</th>
      <th>WDF5</th>
      <th>WSF5</th>
      <th>SNOW</th>
      <th>SNWD</th>
      <th>PRCP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GUAM INTL AP</td>
      <td>GU</td>
      <td>13.4836</td>
      <td>144.7961</td>
      <td>77.4</td>
      <td>20170312</td>
      <td>71.06</td>
      <td>87.08</td>
      <td>80.06</td>
      <td>4.473880</td>
      <td>360.0</td>
      <td>21.027236</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ROOSEVELT ROADS</td>
      <td>PR</td>
      <td>18.2550</td>
      <td>-65.6408</td>
      <td>10.1</td>
      <td>20170404</td>
      <td>77.00</td>
      <td>86.00</td>
      <td>NaN</td>
      <td>8.947760</td>
      <td>360.0</td>
      <td>23.040482</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ROOSEVELT ROADS</td>
      <td>PR</td>
      <td>18.2550</td>
      <td>-65.6408</td>
      <td>10.1</td>
      <td>20170420</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.500372</td>
      <td>360.0</td>
      <td>21.922012</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SAN JUAN L M MARIN AP</td>
      <td>PR</td>
      <td>18.4325</td>
      <td>-66.0108</td>
      <td>2.7</td>
      <td>20170120</td>
      <td>69.08</td>
      <td>82.04</td>
      <td>NaN</td>
      <td>3.355410</td>
      <td>360.0</td>
      <td>17.000744</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SAN JUAN L M MARIN AP</td>
      <td>PR</td>
      <td>18.4325</td>
      <td>-66.0108</td>
      <td>2.7</td>
      <td>20170217</td>
      <td>73.04</td>
      <td>87.08</td>
      <td>NaN</td>
      <td>4.697574</td>
      <td>360.0</td>
      <td>19.908766</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e64bc980-059d-418e-86d1-0b90431328f0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e64bc980-059d-418e-86d1-0b90431328f0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e64bc980-059d-418e-86d1-0b90431328f0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-46533cda-1871-4472-9691-ae7c60e3d3b2">
  <button class="colab-df-quickchart" onclick="quickchart('df-46533cda-1871-4472-9691-ae7c60e3d3b2')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-46533cda-1871-4472-9691-ae7c60e3d3b2 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
'''Filtramos los estados de EE.UU. más visitados por turistas debido a su clima'''
data = data[data['state'].isin(['FL', 'CA', 'HI', 'AZ', 'NV', 'TX', 'SC'])].reset_index(drop=True)
```


```python
'''Verificamos si tenemos datos duplicados'''
print(data.duplicated().sum())
```

    0



```python
# Función para saber la cantidad de datos faltantes por cada variable
def print_null(df):
    print(df.isnull().sum())
print_null(data)
```

    station          0
    state            0
    latitude         0
    longitude        0
    elevation        0
    date             0
    TMIN           301
    TMAX           313
    TAVG         38436
    AWND         19530
    WDF5         22097
    WSF5         22040
    SNOW         31888
    SNWD         11198
    PRCP           298
    dtype: int64



```python
'''Convertimos las columnas de Temperatura a Celsius'''
data['TMIN'] = (data['TMIN'] - 32) * 5/9
data['TMAX'] = (data['TMAX'] - 32) * 5/9
data['TAVG'] = (data['TAVG'] - 32) * 5/9
```


```python
'''Convertimos las siguientes variables de mm a cm'''
data['SNOW'] = data['SNOW'] / 10
data['SNWD'] = data['SNWD'] / 10
data['PRCP'] = data['PRCP'] / 10
```


```python
# Renombrar el nombre de los estados
data['state'] = data['state'].replace({'FL': 'Florida', 'CA': 'California', 'HI': 'Hawái', 'AZ': 'Arizona', 'NV': 'Nevada', 'TX': 'Texas', 'SC': 'Carolina del Sur'})
```


```python
'''Enriquecemos la data con la varaible date'''
# Convertir la columna 'date' a string para poder extraer el mes y día
data['date'] = data['date'].astype(str)

# Crear las nuevas columnas
data['month'] = data['date'].str[4:6].astype(int)  # Dígitos 5 y 6 son el mes
data['day'] = data['date'].str[6:].astype(int)   # Dígitos 7 y 8 son el día

# Eliminamos la variable date
drop_columns = ['date']
data.drop(drop_columns, axis=1, inplace=True)
```


```python
# Creamos un diccionario que mapea los números de mes a los nombres de meses en español
meses_dict = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre'
}

# Aplicar el mapeo a la columna 'mes'
data['month'] = data['month'].map(meses_dict)
```


```python
'''Rellenamos valores nulos de TMIN, TMAX y PRCP con la media'''
# Calcular la media de TMIN y TMAX, ignorando los valores faltantes
mean_TMIN = data['TMIN'].mean()
mean_TMAX = data['TMAX'].mean()
mean_PRCP = data['PRCP'].mean()

# Imputar los valores faltantes en TMIN, TMAX y PRCP con su respectiva media
data['TMIN'].fillna(mean_TMIN, inplace=True)
data['TMAX'].fillna(mean_TMAX, inplace=True)
data['PRCP'].fillna(mean_PRCP, inplace=True)
```

#### **3.2 Nivel II: Análisis de Outliers**


```python
# Calcular Q1 (percentil 25) y Q3 (percentil 75) para cada variable
Q1 = data[['TMIN', 'TMAX', 'TAVG', 'AWND', 'WDF5', 'WSF5', 'SNOW', 'SNWD', 'PRCP']].quantile(0.25)
Q3 = data[['TMIN', 'TMAX', 'TAVG', 'AWND', 'WDF5', 'WSF5', 'SNOW', 'SNWD', 'PRCP']].quantile(0.75)
IQR = Q3 - Q1

# Definir los límites de atipicidad
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detectar valores fuera de estos límites
outliers = (data[['TMIN', 'TMAX', 'TAVG', 'AWND', 'WDF5', 'WSF5', 'SNOW', 'SNWD', 'PRCP']] < lower_bound) | (data[['TMIN', 'TMAX', 'TAVG', 'AWND', 'WDF5', 'WSF5', 'SNOW', 'SNWD', 'PRCP']] > upper_bound)
```


```python
# Reemplazar valores atípicos con la mediana
for col in ['TMIN', 'TMAX', 'TAVG', 'AWND', 'WDF5', 'WSF5', 'SNOW', 'SNWD', 'PRCP']:
    median = data[col].mean()  # Calcular la media de la columna
    data.loc[outliers[col], col] = median  # Reemplazar valores atípicos con la mediana
```


```python
'''Visualización luego de imputar los valores atípicos'''
plt.figure(figsize=(15, 8))
sns.boxplot(data=data[['TMIN', 'TMAX', 'TAVG', 'AWND', 'WDF5', 'WSF5', 'SNOW', 'SNWD', 'PRCP']])
plt.show()
```

#### **3.2 Nivel III: Análisis de Valores Faltantes**

- En esta sección imputamos valores faltantes en diferentes columnas meteorológicas `(AWND, WDF5, WSF5, SNOW, y SNWD)` dentro de nuestro dataset. Para cada columna con valores faltantes, el proceso es similar: se divide el dataset en datos completos y faltantes, se definen variables predictoras y la variable objetivo, y las variables categóricas como state y month se codifican usando LeaveOneOutEncoder. Luego, se entrena un modelo de regresión basado en XGBRegressor, aplicando Recursive Feature Elimination `(RFE)` para seleccionar las características más relevantes. El modelo se evalúa utilizando métricas como el error cuadrático medio `(MSE)` y el coeficiente de determinación `(R²)`, además de validación cruzada para medir su desempeño. Finalmente, el modelo entrenado se utiliza para predecir los valores faltantes, que se imputan de manera segura en el dataset original.


```python
# Dividimos el dataset en datos completos y faltantes en TAVG
data_complete = data[data['TAVG'].notna()]
data_missing = data[data['TAVG'].isna()]

# Variables predictoras y variable objetivo
features = ['state', 'TMIN', 'TMAX', 'month', 'day', 'elevation']
target = 'TAVG'

# Codificamos las variables categóricas
encoder = LeaveOneOutEncoder(cols=['state', 'month'])
data_complete_encoded = encoder.fit_transform(data_complete[features], data_complete[target])
data_missing_encoded = encoder.transform(data_missing[features])

# Separamos las características y el objetivo
X_complete = data_complete_encoded[features]
y_complete = data_complete[target]

# Dividimos el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

# Creamos el modelo XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# Aplicamos RFE para seleccionar las características más importantes
selector = RFE(model, n_features_to_select=3, step=1)
selector = selector.fit(X_train, y_train)

# Entrenamos el modelo con las características seleccionadas
model.fit(X_train.iloc[:, selector.support_], y_train)

# Evaluamos el modelo
y_pred = model.predict(X_test.iloc[:, selector.support_])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"Características seleccionadas (True indica que la característica fue seleccionada): {selector.support_}")

# Imputamos los valores faltantes
X_missing = data_missing_encoded[features].iloc[:, selector.support_]
predictions = model.predict(X_missing)

# Actualizamos el DataFrame original con los valores imputados de forma segura
data.loc[data['TAVG'].isna(), 'TAVG'] = predictions
```

    MSE: 0.6815625678390964
    R2: 0.993595371878675
    Características seleccionadas (True indica que la característica fue seleccionada): [ True  True  True False False False]



```python
# Dividimos el dataset en datos completos y faltantes en TAVG
data_complete = data[data['AWND'].notna()]
data_missing = data[data['AWND'].isna()]

# Variables predictoras y variable objetivo
features = ['state', 'elevation', 'month', 'day', 'WDF5', 'WSF5']
target = 'AWND'

# Codificamos las variables categóricas
encoder = LeaveOneOutEncoder(cols=['state', 'month'])
data_complete_encoded = encoder.fit_transform(data_complete[features], data_complete[target])
data_missing_encoded = encoder.transform(data_missing[features])

# Separamos las características y el objetivo
X_complete = data_complete_encoded[features]
y_complete = data_complete[target]

# Dividimos el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

# Creamos el modelo XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# Aplicamos RFE para seleccionar las características más importantes
selector = RFE(model, n_features_to_select=3, step=1)
selector = selector.fit(X_train, y_train)

# Entrenamos el modelo con las características seleccionadas
model.fit(X_train.iloc[:, selector.support_], y_train)

# Evaluamos el modelo
y_pred = model.predict(X_test.iloc[:, selector.support_])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"Características seleccionadas (True indica que la característica fue seleccionada): {selector.support_}")

# Imputamos los valores faltantes
X_missing = data_missing_encoded[features].iloc[:, selector.support_]
predictions = model.predict(X_missing)

# Actualizamos el DataFrame original con los valores imputados de forma segura
data.loc[data['AWND'].isna(), 'AWND'] = predictions
```

    MSE: 0.450116279359787
    R2: 0.9707041473088335
    Características seleccionadas (True indica que la característica fue seleccionada): [ True False  True False False  True]



```python
# Dividimos el dataset en datos completos y faltantes en WDF5
data_complete = data[data['WDF5'].notna()]
data_missing = data[data['WDF5'].isna()]

# Variables predictoras y variable objetivo
features = ['state', 'elevation', 'month', 'day', 'AWND', 'WSF5']
target = 'WDF5'

# Codificamos las variables categóricas
encoder = LeaveOneOutEncoder(cols=['state', 'month'])
data_complete_encoded = encoder.fit_transform(data_complete[features], data_complete[target])
data_missing_encoded = encoder.transform(data_missing[features])

# Separamos las características y el objetivo
X_complete = data_complete_encoded[features]
y_complete = data_complete[target]

# Dividimos el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

# Creamos el modelo XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# Aplicamos RFE para seleccionar las características más importantes
selector = RFE(model, n_features_to_select=3, step=1)
selector = selector.fit(X_train, y_train)

# Entrenamos el modelo con las características seleccionadas
model.fit(X_train.iloc[:, selector.support_], y_train)

# Evaluamos el modelo
y_pred = model.predict(X_test.iloc[:, selector.support_])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"Características seleccionadas (True indica que la característica fue seleccionada): {selector.support_}")

# Imputamos los valores faltantes
X_missing = data_missing_encoded[features].iloc[:, selector.support_]
predictions = model.predict(X_missing)

# Actualizamos el DataFrame original con los valores imputados de forma segura
data.loc[data['WDF5'].isna(), 'WDF5'] = predictions
```

    MSE: 3.7725316894786802
    R2: 0.9995713741986438
    Características seleccionadas (True indica que la característica fue seleccionada): [ True  True  True False False False]



```python
# Dividimos el dataset en datos completos y faltantes en TAVG
data_complete = data[data['WSF5'].notna()]
data_missing = data[data['WSF5'].isna()]

# Variables predictoras y variable objetivo
features = ['state', 'elevation', 'month', 'day', 'AWND', 'WDF5']
target = 'WSF5'

# Codificamos las variables categóricas
encoder = LeaveOneOutEncoder(cols=['state', 'month'])
data_complete_encoded = encoder.fit_transform(data_complete[features], data_complete[target])
data_missing_encoded = encoder.transform(data_missing[features])

# Separamos las características y el objetivo
X_complete = data_complete_encoded[features]
y_complete = data_complete[target]

# Dividimos el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

# Creamos el modelo XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# Aplicamos RFE para seleccionar las características más importantes
selector = RFE(model, n_features_to_select=3, step=1)
selector = selector.fit(X_train, y_train)

# Entrenamos el modelo con las características seleccionadas
model.fit(X_train.iloc[:, selector.support_], y_train)

# Evaluamos el modelo
y_pred = model.predict(X_test.iloc[:, selector.support_])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"Características seleccionadas (True indica que la característica fue seleccionada): {selector.support_}")

# Imputamos los valores faltantes
X_missing = data_missing_encoded[features].iloc[:, selector.support_]
predictions = model.predict(X_missing)

# Actualizamos el DataFrame original con los valores imputados de forma segura
data.loc[data['WSF5'].isna(), 'WSF5'] = predictions
```

    MSE: 2.711701529936648
    R2: 0.9614981031592947
    Características seleccionadas (True indica que la característica fue seleccionada): [ True  True False False  True False]



```python
# Dividimos el dataset en datos completos y faltantes en SNOW
data_complete = data[data['SNOW'].notna()]
data_missing = data[data['SNOW'].isna()]

# Variables predictoras y variable objetivo
features = ['state', 'TAVG', 'month', 'PRCP', 'WSF5', 'elevation']
target = 'SNOW'

# Codificamos las variables categóricas
encoder = LeaveOneOutEncoder(cols=['state', 'month'])
data_complete_encoded = encoder.fit_transform(data_complete[features], data_complete[target])
data_missing_encoded = encoder.transform(data_missing[features])

# Separamos las características y el objetivo
X_complete = data_complete_encoded
y_complete = data_complete[target]

# Dividimos el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

# Creamos el modelo XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# Aplicamos RFE para seleccionar las características más importantes
selector = RFE(model, n_features_to_select=4, step=1)
selector.fit(X_train, y_train)

# Entrenamos el modelo con las características seleccionadas
model.fit(X_train.iloc[:, selector.support_], y_train)

# Evaluamos el modelo
y_pred = model.predict(X_test.iloc[:, selector.support_])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"Características seleccionadas (True indica que la característica fue seleccionada): {selector.support_}")

# Imputamos los valores faltantes
X_missing = data_missing_encoded[features].iloc[:, selector.support_]
predictions = model.predict(X_missing)

# Actualizamos el DataFrame original con los valores imputados de forma segura
data.loc[data['SNOW'].isna(), 'SNOW'] = predictions
```

    MSE: 7.619655805268205e-05
    R2: 0.8519219747260094
    Características seleccionadas (True indica que la característica fue seleccionada): [ True  True  True  True False False]



```python
# Dividimos el dataset en datos completos y faltantes en TAVG
data_complete = data[data['SNWD'].notna()]
data_missing = data[data['SNWD'].isna()]

# Variables predictoras y variable objetivo
features = ['state', 'TMIN', 'month', 'day', 'WSF5', 'SNOW']
target = 'SNWD'

# Codificamos las variables categóricas
encoder = LeaveOneOutEncoder(cols=['state', 'month'])
data_complete_encoded = encoder.fit_transform(data_complete[features], data_complete[target])
data_missing_encoded = encoder.transform(data_missing[features])

# Separamos las características y el objetivo
X_complete = data_complete_encoded[features]
y_complete = data_complete[target]

# Dividimos el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

# Creamos el modelo XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# Aplicamos RFE para seleccionar las características más importantes
selector = RFE(model, n_features_to_select=3, step=1)
selector = selector.fit(X_train, y_train)

# Entrenamos el modelo con las características seleccionadas
model.fit(X_train.iloc[:, selector.support_], y_train)

# Evaluamos el modelo
y_pred = model.predict(X_test.iloc[:, selector.support_])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"Características seleccionadas (True indica que la característica fue seleccionada): {selector.support_}")

# Imputamos los valores faltantes
X_missing = data_missing_encoded[features].iloc[:, selector.support_]
predictions = model.predict(X_missing)

# Actualizamos el DataFrame original con los valores imputados de forma segura
data.loc[data['SNWD'].isna(), 'SNWD'] = predictions
```

    MSE: 0.25330630380117813
    R2: 0.9718307776170203
    Características seleccionadas (True indica que la característica fue seleccionada): [ True  True False False  True False]



```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 76351 entries, 0 to 76350
    Data columns (total 16 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   station    76351 non-null  object 
     1   state      76351 non-null  object 
     2   latitude   76351 non-null  float64
     3   longitude  76351 non-null  float64
     4   elevation  76351 non-null  float64
     5   TMIN       76351 non-null  float64
     6   TMAX       76351 non-null  float64
     7   TAVG       76351 non-null  float64
     8   AWND       76351 non-null  float64
     9   WDF5       76351 non-null  float64
     10  WSF5       76351 non-null  float64
     11  SNOW       76351 non-null  float64
     12  SNWD       76351 non-null  float64
     13  PRCP       76351 non-null  float64
     14  month      76351 non-null  object 
     15  day        76351 non-null  int64  
    dtypes: float64(12), int64(1), object(3)
    memory usage: 9.3+ MB


### **4. Insights**

**Insight 1: Comparación de las temperaturas promedio mensuales y el turismo en los 7 estados**

- **Hipótesis:** Los meses con temperaturas promedio más agradables tienden a atraer más turistas en todos los estados.

- **Pregunta exploratoria:** ¿En qué meses se registran las temperaturas promedio más agradables en cada estado, y cómo se correlaciona esto con el número de turistas?

- **Motivación:** Analizar las temperaturas promedio en diferentes meses puede ayudarte a identificar los meses en los que las condiciones meteorológicas son óptimas para el turismo en cada estado.


```python
plt.rc('font', family='serif')

month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre']

data['month'] = pd.Categorical(data['month'], categories=month_order, ordered=True)

monthly_avg_temp = data.groupby(['state', 'month'], observed=True)['TAVG'].mean().reset_index()

plt.figure(figsize=(14, 8))
sns.lineplot(data=monthly_avg_temp, x='month', y='TAVG', hue='state', marker='o', palette='tab10', linewidth=2.5)

plt.title('Temperatura Promedio Mensual por Estado', fontsize=18, fontweight='bold')
plt.xlabel('Mes', fontsize=14)
plt.ylabel('Temperatura Promedio (°C)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Estado', title_fontsize='13', fontsize='11')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```


    
![png](https://github.com/JosueVelo/Guia_4/blob/main/N1.png?raw=true)
    


**Insight 2: Evolución de la precipitación mensual y su impacto en el turismo en Florida y Hawái**

- **Hipótesis:** Los meses con menos precipitación tienden a mostrar un aumento en el número de turistas, especialmente en estados como Florida y Hawái que tienen temporadas de lluvias.

- **Pregunta exploratoria:** ¿Cómo varía la precipitación mensual en Florida y Hawái a lo largo del año y cómo puede afectar al número de turistas en esos meses?

- **Motivación:** Entender la relación entre la precipitación y el turismo puede ayudarte a prever los meses en que la falta de lluvia puede atraer a más visitantes.


```python
estados_interes = ['Florida', 'Hawái']
data_filtrada = data[data['state'].isin(estados_interes)]

promedio_mensual_prcp = data_filtrada.groupby(['state', 'month'], observed=True)['PRCP'].mean().reset_index()

plt.figure(figsize=(14, 8))
sns.lineplot(data=promedio_mensual_prcp, x='month', y='PRCP', hue='state', marker='o', palette='viridis', linewidth=2.5)

plt.title('Evolución de la Precipitación Promedio Mensual en Florida y Hawái', fontsize=18, fontweight='bold')
plt.xlabel('Mes', fontsize=14)
plt.ylabel('Precipitación Promedio (cm)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Estado', title_fontsize='13', fontsize='11')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```


    
![png]([output_34_0.png](https://github.com/JosueVelo/Guia_4/blob/main/N2.png?raw=true))
    


**Insight 3: Relación entre la velocidad del viento y el turismo en los estados del suroeste (Nevada y Arizona)**

- **Hipótesis:** Los meses con velocidades de viento más bajas podrían ser más favorables para el turismo en estados como Nevada y Arizona, que tienen climas áridos.

- **Pregunta exploratoria:** ¿En qué meses la velocidad del viento es más baja en Nevada y Arizona, y cómo se relaciona esto con el aumento en el número de turistas?

- **Motivación:** Analizar la velocidad del viento en relación con el turismo puede proporcionar información sobre las condiciones meteorológicas ideales para los visitantes en estados áridos.


```python
estados_suroeste = ['Nevada', 'Arizona']
data_filtrada = data[data['state'].isin(estados_suroeste)]

promedio_mensual_viento = data_filtrada.groupby(['state', 'month'], observed=True)['AWND'].mean().unstack()

plt.figure(figsize=(10, 6))
sns.heatmap(promedio_mensual_viento, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5, cbar_kws={'label': 'Velocidad Promedio del Viento (m/s)'})

plt.title('Velocidad Promedio del Viento Mensual en Nevada y Arizona', fontsize=13, fontweight='bold')
plt.xlabel('Mes', fontsize=14)
plt.ylabel('Estado', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
```


    
![png](https://github.com/JosueVelo/Guia_4/blob/main/N3.png?raw=true)
    


**Insight 4:  Impacto de la nieve y la temperatura mínima en el turismo en Arizona**

- **Hipótesis:** Los meses con mayor profundidad de nieve en Arizona podrían coincidir con temperaturas mínimas más bajas, lo que podría atraer a más turistas interesados en actividades invernales. En cambio, en meses con temperaturas mínimas más altas, la profundidad de nieve podría ser menor, afectando el interés en actividades de nieve.

- **Pregunta exploratoria:** ¿Cómo afecta la profundidad de nieve y la temperatura mínima en Arizona al número de turistas durante los meses del año?

- **Motivación:** Este análisis puede ayudar a identificar los meses en los que las condiciones de nieve y la temperatura mínima podrían influir en las decisiones de viaje hacia Arizona. Entender esta relación puede ser útil para estrategias de marketing y planificación de actividades turísticas.


```python
estado_interes = 'Arizona'
data_filtrada_arizona = data[data['state'] == estado_interes]

promedio_mensual_nieve_tmin = data_filtrada_arizona.groupby(['month'], observed=True).agg({'SNWD': 'mean', 'TMIN': 'mean'}).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 8))

sns.barplot(data=promedio_mensual_nieve_tmin, x='month', y='SNWD', color='b', alpha=0.7, ax=ax1)

for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x() + p.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black')

ax2 = ax1.twinx()
sns.lineplot(data=promedio_mensual_nieve_tmin, x='month', y='TMIN', marker='o', color='r', linestyle='--', ax=ax2)

ax1.set_title('Profundidad Promedio de Nieve y Temperatura Mínima Mensual en Arizona', fontsize=16, fontweight='bold')
ax1.set_xlabel('Mes', fontsize=12)
ax1.set_ylabel('Profundidad de Nieve (cm)', fontsize=12)
ax2.set_ylabel('Temperatura Mínima Promedio (°C)', fontsize=12, color='r')
ax1.legend(['Profundidad de Nieve'], loc='upper left', fontsize='11')
ax2.legend(['Temperatura Mínima'], loc='upper right', fontsize='11')
ax1.grid(True, linestyle='--', alpha=0.7)
ax2.grid(False)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


    
![png](https://github.com/JosueVelo/Guia_4/blob/main/N4.png?raw=true)
    


**Insight 5: Análisis de la Temperatura Promedio Diaria en Junio y su Impacto en el Turismo en Texas y California**

- **Hipótesis:** Las temperaturas promedio `(TAVG)` en el mes de junio en California son más moderadas que en Texas, lo que puede influir en la elección de los turistas al planificar sus viajes de verano.

- **Pregunta exploratoria:** ¿Cómo varían las temperaturas promedio diarias en junio entre Texas y California, y cómo estas diferencias pueden afectar la afluencia turística?

- **Motivación:** El mes de junio marca el inicio de la temporada de verano, lo que es crucial para el turismo. Identificar si Texas presenta temperaturas más extremas que California en este mes permitirá analizar la comodidad climática para los turistas y su impacto en el turismo. California, con su clima más templado, podría atraer a más visitantes en comparación con Texas, donde las temperaturas pueden ser más altas y menos favorables para actividades al aire libre.


```python
estados_interes = ['Texas', 'California']

data_filtrada = data[(data['state'].isin(estados_interes)) & (data['month'] == 'Junio')]

promedio_diario_tavg = data_filtrada.groupby(['state', 'day'], observed=True)['TAVG'].mean().reset_index()

plt.figure(figsize=(14, 8))
sns.lineplot(data=promedio_diario_tavg, x='day', y='TAVG', hue='state', marker='o', palette='viridis', linewidth=2.5)

plt.title('Variación de la Temperatura Promedio Diaria de Junio en Texas y California', fontsize=18, fontweight='bold')
plt.xlabel('Día del Mes', fontsize=14)
plt.ylabel('Temperatura Promedio (°C)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Estado', title_fontsize='13', fontsize='11')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```


    
![png](https://github.com/JosueVelo/Guia_4/blob/main/N5.png?raw=true)
    


### **Diccionario de Datos**

1. **station:** Nombre único para cada estación meteorológica.

2. **state:** Ubicación geográfica del estado de EE. UU. donde se encuentra la estación meteorológica.

3. **latitude:** Latitud de la ubicación de la estación meteorológica, expresada en grados decimales.

4. **longitude:** Longitud de la ubicación de la estación meteorológica, expresada en grados decimales.

5. **elevation:** Altitud de la estación meteorológica en metros sobre el nivel del mar.

6. **TMIN:** Temperatura mínima diaria, en grados Celsius.

7. **TMAX:** Temperatura máxima diaria, en grados Celsius.

8. **TAVG:** Temperatura promedio diaria, en grados Celsius.

9. **AWND:** Velocidad promedio diaria del viento, en metros por segundo.

10. **WDF5:** Dirección del viento más fuerte durante 5 segundos, en grados.

11. **WSF5:** Velocidad del viento más fuerte durante 5 segundos, en metros por segundo.

12. **SNOW:** Cantidad de nieve caída durante el día, en centímetros.

13. **SNWD:** Profundidad del manto de nieve en el suelo, en centímetros.

14. **PRCP:**  Precipitación total diaria, en centímetros.

15. **día:** Día en que se realizó la observación meteorológica.

16. **mes:** Mes en que se realizó la observación meteorológica.

### **4. Bibliografía**

1. https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

2. https://contrib.scikit-learn.org/category_encoders/leaveoneout.html

3. https://www.geeksforgeeks.org/xgboost-for-regression/

4. https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
