import re

import pandas as pd
import numpy as np


def corrige_colunas(col_name):
    """
    :param col_name: Nome da coluna a ser tratada/formatada
    :return: Fará o tratamento do nome das colunas informadas para evitar problemas no consumo dos dados
    """
    return re.sub(r"/| ", "", col_name).lower()


def taxa_cresimento(data, variable, data_inicio=None, data_fim=None):
    """
    Fórmula para Taxa de Cresimento: (presente/passado) ** (1/n) - 1
    :param data: Dados para cálculo
    :param variable: Variável
    :param data_inicio: (Opcional) Se for "None", será definida como a primeira data disponível
    :param data_fim: (Opcional) Se for "None", será definida como a última data disponível
    :return: Taxa de crescimento
    """
    if data_inicio is None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)

    if data_fim is None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)

    # Define os valores do presente e passado
    passado = data.loc[data.observationdate == data_inicio, variable].values[0]
    presente = data.loc[data.observationdate == data_fim, variable].values[0]

    # Define o número de pontos no tempo avaliado
    n = (data_fim - data_inicio).days

    # Cálculo da Taxa
    taxa = (presente / passado) ** (1 / n) - 1

    return taxa * 100


def taxa_cresimento_diaria(data, variable, data_inicio=None):
    """
        Fórmula para Taxa de Cresimento: (presente/passado) ** (1/n) - 1
        :param data: Dados para cálculo
        :param variable: Variável
        :param data_inicio: (Opcional) Se for "None", será definida como a primeira data disponível
        :return:
        """
    if data_inicio is None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)

    data_fim = data.observationdate.max()
    # Define o número de pontos no tempo avaliado
    n = (data_fim - data_inicio).days

    # Cálculo da Taxa de um dia para o outro
    taxas = list(map(
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x - 1]) / data[variable].iloc[x - 1],
        range(1, n + 1)
    ))
    return np.array(taxas) * 100



