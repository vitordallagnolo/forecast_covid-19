import utils as u
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pmdarima 
from datetime import datetime
from fbprophet import Prophet
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose


url = 'https://raw.githubusercontent.com/vitordallagnolo/forecast_covid-19/main/covid_19_data.csv'

df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update'])

print(df.dtypes)

df.columns = [u.corrige_colunas(col) for col in df.columns]

df.loc[df.countryregion == 'Brazil']

brasil = df.loc[
    (df.countryregion == 'Brazil') &
    (df.confirmed > 0)
    ]

px.line(brasil, 'observationdate', 'confirmed', title='Casos confirmados no Brasil')
fig = px.line(brasil, 'observationdate', 'confirmed', title='Casos confirmados no Brasil')
fig.write_html("file.html")

brasil['novoscasos'] = list(map(
    lambda x: 0 if (x == 0) else brasil['confirmed'].iloc[x] - brasil['confirmed'].iloc[x - 1],
    np.arange(brasil.shape[0])
))

fig = px.line(brasil, 'observationdate', 'novoscasos', title='Novos Casos por dia no Brasil')
fig.write_html("novoscasos.html")

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=brasil.observationdate, y=brasil.deaths, name='Mortes',
               mode='lines+markers', line={'color': 'red'})
)

fig.update_layout(title='Mortes por COVID-19 no Brasil')
fig.write_html("mortes.html")
tx_geral = u.taxa_cresimento(brasil, 'confirmed')
print("Taxa de crescimento geral de casos confimados")
print(tx_geral)

tx_dia = u.taxa_cresimento_diaria(brasil, 'confirmed')
print("Taxa de crescimento diário de casos confimados")
print(tx_dia)

primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()
px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1:],
        y=tx_dia, title='Taxa de crescimento de casos confirmados no Brasil')

fig = px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1:],
              y=tx_dia, title='Taxa de crescimento de casos confirmados no Brasil')
fig.write_html("crescimento_diario.html")

confirmados = brasil.confirmed
confirmados.index = brasil.observationdate
print(confirmados)

res = seasonal_decompose(confirmados)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))

ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(confirmados.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()

modelo = auto_arima(confirmados)

fig = go.Figure(go.Scatter(
    x=confirmados.index,
    y=confirmados,
    name='Observados'
))

fig.add_trace(go.Scatter(
    x=confirmados.index,
    y=modelo.predict_in_sample(),
    name='Preditos'
))

fig.add_trace(go.Scatter(
    x=pd.date_range('2020-05-20', '2020-06-20'),
    y=modelo.predict(31),
    name='Forecast'
))

fig.update_layout(title='Previsão de casos confirmados no Brasil para os próximos 30 dias')
fig.write_html("previsao.html")
#fig.show()

# Preprocessamentos
train = confirmados.reset_index()[:-5]
test = confirmados.reset_index()[-5:]

# Renomeando colunas
train.rename(columns={'observationdate': 'ds', 'confirmed': 'y'}, inplace=True)
test.rename(columns={'observationdate': 'ds', 'confirmed': 'y'}, inplace=True)

# Definir o modelo de crescimento
profeta = Prophet(growth='logistic', changepoints=['2020-03-21', '2020-03-30',
                                                   '2020-04-25', '2020-05-03',
                                                   '2020-05-10'])

pop = 211463256
train['cap'] = pop

# Treinar o modelo
profeta.fit(train)

# Construir previsões para o futuro
future_dates = profeta.make_future_dataframe(periods=200)
future_dates['cap'] = pop
forecast = profeta.predict(future_dates)

fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predição'))
# fig.add_trace(go.Scatter(x=test.index, y=test, name='Observados - Teste'))
fig.add_trace(go.Scatter(x=train.ds, y=train.y, name='Observados - Treino'))
fig.update_layout(title='Predições de casos confirmados no Brasil')
fig.write_html("predict_long.html")

