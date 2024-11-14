import pandas as pd
import glob
from pathlib import Path

def df_union(directoty, extradel = 1):
    dir_path = Path(directory)
    files = [i for i in dir_path.rglob('*.csv')]
    #Создаем пустой DataFrame, который будет содержать данные из всех файлов
    df_list = []

    # Цикл по всем найденным файлам
    for file in files:
        # Читаем файл и добавляем столбец с именем директории
        df_part = pd.read_csv(file, sep=';', header=0)
        df_part['Directory'] = file.name  # Имя директории, в которой находится файл
        df_list.append(df_part)

    df = pd.concat(df_list, ignore_index=True)
    def data_base_mod(df):
        df = df[df['Дата выхода'] != 'Открыта']
        df = df[['Позиция', 'Символ', 'Дата входа', 'Время входа', 'Дата выхода', 'Время выхода', 'П/У сделки', 'Продолж. (баров)', 'MAE %', 'MFE %', '% изменения', 'Directory']]
        df.loc[:, 'Позиция'] =np.where(df.Позиция == 'Длинная', 1, -1)
        df.loc[:,'date_time_in'] = pd.to_datetime(df['Дата входа'].str.strip() + ' ' + df['Время входа'].str.strip(), format='%d.%m.%Y %H:%M:%S', dayfirst=True)
        df.loc[:,'date_time_out'] = pd.to_datetime(df['Дата выхода'].str.strip() + ' ' + df['Время выхода'].str.strip(), format='%d.%m.%Y %H:%M:%S', dayfirst=True)
        df = df.rename(columns={'Позиция': 'naprav', 'П/У сделки': 'result', 'Продолж. (баров)': 'duration', 'Символ': 'symbol', '% изменения': 'result_%'})
        df.drop(['Дата входа', 'Время входа', 'Дата выхода', 'Время выхода'], axis=1, inplace=True)
        df = df[df['date_time_out'].notnull()]
        df['duration'] = (df['date_time_out'] - df['date_time_in']).dt.total_seconds() / 60
        df['duration'] = df['duration'].astype('int')
        df['result'] = df['result'].str.replace(' ', '').astype('float')
        df['MAE %'] = df['MAE %'].str.rstrip(' %').astype('float') / 100
        df['MFE %'] = df['MFE %'].str.rstrip(' %').astype('float') / 100
        df['result_%'] = df['result_%'].astype('float') / 100
        df['date_out'] = df['date_time_out'].dt.date
        return df
    
    df = data_base_mod(df)
    if extradel == 1:
        # Удаляем аномальный результат в VTBR
        upper_bound = df['result'].quantile(0.999)
        df = df[~(df['result']>upper_bound)]
        lower_bound = df['result'].quantile(0.001)
        df = df[~(df['result']<lower_bound)]
    return df

# следующий модуль - отрисовка общей эквити
def union_chart(df, sostav = 'naprav', col_id=0):
    
    df_pivot = df1.pivot_table(index='date_out',\
                                 columns=sostav,\
                                 values='result',\
                                 aggfunc='sum')
    df_pivot['tot']=df_pivot.sum(axis=1)
    
    df_pivot = df_pivot.apply(lambda x: x.fillna(0).cumsum()) # делаем нарастающий
    col = df_pivot.columns # сохраняем названия колонок в переменную
    df_pivot.index.name = 'date_out'
    df_pivot.columns.name = None
    df_pivot = df_pivot.reset_index()
    df_pivot.set_index('date_out', inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if col_id == 0:
        col = 'tot'
    ax.plot(df_pivot.index, df_pivot[col], label=col)
    # _=ax.xticks(rotation=90)
    ax.legend(loc='upper left')
