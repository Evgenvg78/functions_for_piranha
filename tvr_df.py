
import pandas as pd
from moexalgo import session, Market, Ticker, CandlePeriod
import numpy as np
import csv


def TVR_transform(TVR, varyant_types, type='Go'):
  if type == 'Go':
      col_count = 'INITIALMARGIN'
  else:
      col_count = 'full_price'

  # модуль получения данных с биржи
  # загрузка данных по инструментам !!!!!! не работает вне торговой сессии !!!!

  # forts_tickers = Market('FO').tickers()
  # sec_data_list = []
  # for i in range(len(forts_tickers)):
  #   # print(i)
  #   name = forts_tickers.iloc[i, 0]
  #   sec = Ticker(name).info()
  #   sec = sec.T
  #   sec_data_list.append(sec)
  sec_data_file = '/content/drive/MyDrive/work_data/TVR/sec_tvr.csv'
  sec_data = pd.read_csv(sec_data_file, sep=',', index_col=False)
  # sec_data = pd.concat(sec_data_list, axis=0).reset_index(drop=True)
  sec_tvr = sec_data[['SECID', 'MINSTEP', 'STEPPRICE', 'PREVSETTLEPRICE', 'INITIALMARGIN', 'BUYSELLFEE', 'SCALPERFEE']]
  sec_tvr['full_price'] = sec_tvr['PREVSETTLEPRICE'] / sec_tvr['MINSTEP'] * sec_tvr['STEPPRICE']
  
  

  

  # функция преобразования 3-5 в лист [3, 4, 5]

  def range_num_to_list(range_num):

    result = []
    for element in range_num.split(','):
      split_index = element.find('-')
      if split_index == -1:
        result.append(int(element))
      else:
        start = int(element[:split_index])
        end = int(element[split_index+1:])
        result.extend(list(range(start, end+1)))

    return result
  
  # модуль преобразования типа скриптов

  varyant_types = pd.read_csv(varyant_types, sep='=', header=0)
  varyant_types['variant'] = varyant_types['variant'].apply(range_num_to_list)
  varyant_types = varyant_types.explode('variant')


  # основной модуль


  with open(TVR, encoding='utf8') as file:
      separator = file.readline()[0]  # Получаем элемент, который разделяет названия колонок
  with open(TVR, encoding='utf8') as file:
      column_names = file.readline().strip().split(separator)[2:]  # Получаем названия колонок начиная с третьего элемента
  data = []
  with open(TVR, encoding='utf8') as file:
      lines = file.readlines()[1:]  # Пропускаем первую строку с заголовками
      reader = csv.reader(lines, delimiter=' ')
      for row in reader:
          if len(row) > 2:
              row[2] = ' '.join(row[2:])  # Объединяем оставшиеся элементы в третий столбец
              data.append(row[:3])  # Добавляем только первые три элемента в список

  # Преобразуем список в DataFrame
  df = pd.DataFrame(data, columns=['stroka', 'stolbec', 'data'])

  # Изменяем тип данных на целочисленные
  df['stroka'] = df['stroka'].astype(int)
  df['stolbec'] = df['stolbec'].astype(int)

  # создаем паттерн, чтобы все стобцы были учтены
  stroka = [9999]*56
  stolbec = list(range(1,57))
  data = ['**']*56
  pattern_dic = {'stroka': stroka, 'stolbec': stolbec, 'data':data}
  pattern_df = pd.DataFrame(pattern_dic)

  df = pd.concat([df,pattern_df])
  df = df.replace(r'^\s*$', np.nan, regex=True)

  tvr_table = df.pivot(index='stroka', columns='stolbec', values='data') # разворачиваем таблицу
  tvr_table.columns = column_names
  tvr_table.reset_index(inplace = True)
  tvr_with_types = tvr_table

  # tvr_with_types = tvr_table.merge(varyant_types, how = 'left', left_on='stroka', right_on='variant') # присоединяем типы скриптов
  tvr_with_types['stroka'] = tvr_with_types['stroka'].astype(int)

  condition = (tvr_with_types['Start']=='True') & (tvr_with_types['Kill all']!='True') #& (tvr_with_types['Out only']!='True') # фильтр - оставляем только действующих роботов
  tvr_with_types = tvr_with_types[condition] # применяем предыдущий фильрт

  # вставляем новый столбец np.nan
  stolbec_ryadom = 'V 0'
  index_stolbec = tvr_with_types.columns.get_loc(stolbec_ryadom)
  tvr_with_types.insert(index_stolbec+1, 'W 0', np.nan)

  col_names = tvr_with_types.columns.tolist() # сохраняем старые названия колонок
  tvr_with_types = tvr_with_types.reset_index()

  # это для того, чтобы потом сделать unpivot
  col_n = list(range(1,len(tvr_with_types.columns))) # создаем нумерованный лист для названия колонок
  col_n.insert(0, 'stroka') # в первую позицию поставим имя 'stroka'
  tvr_with_types.columns = col_n

  tvr_with_types = pd.melt(tvr_with_types, id_vars=['stroka'], var_name='stolbec', value_name='data') # делаем "обратный процесс
  # анпивот, чтобы провести манипуляции с вставкой информации по инструменту - ГО


  tvr_with_types = tvr_with_types.sort_values(by=['stroka', 'stolbec'], ascending=True)
  

  tvr_with_types = tvr_with_types.merge(sec_tvr, how='left', left_on='data', right_on='SECID') # присоединяем таблицу с ГО
  tvr_with_types[col_count] = tvr_with_types[col_count].shift(2) # передвигаем цену ниже на нужную строку
  tvr_with_types['data'] = np.where(tvr_with_types[col_count] > 0, tvr_with_types[col_count], tvr_with_types['data']) # вставляем ГО в структуру

  tvr_with_types = tvr_with_types.iloc[:, 0:3] #оставляем только нужные колонки

  tvr_with_types = tvr_with_types.pivot(index='stroka', columns='stolbec', values='data')
  tvr_with_types.columns = col_names

  tvr_with_types.reset_index(drop = True, inplace = True)
  tvr_with_types = tvr_with_types.sort_values(by=['Mode', 'stroka'])
  tvr_with_types['indexxx'] = tvr_with_types.groupby('Mode').cumcount()
  tvr_with_types = tvr_with_types.sort_values(by=['indexxx', 'stroka'])

  tvr_with_types['varr1'] = np.where(tvr_with_types['Mode']=='1', tvr_with_types['stroka'].astype(str), np.nan)
  tvr_with_types['varr2'] = np.where(tvr_with_types['Mode']=='-1', tvr_with_types['stroka'].astype(str), np.nan)
  tvr_with_types['varr1'].ffill(inplace = True)
  tvr_with_types['varr2'].bfill(inplace = True)
  tvr_with_types['variant'] = tvr_with_types['varr1'].astype(str)+'-'+tvr_with_types['varr2'].astype(str)
  tvr_with_types.drop(columns=['indexxx', 'varr1', 'varr2'], inplace = True)

  # считаем ГО
  # Извлечение  имен  столбцов
  V_cols = [col for col in tvr_with_types.columns if col.startswith('V')]
  W_cols = [col for col in tvr_with_types.columns if col.startswith('W')]

  # Проверка, что количество столбцов "V" и "W" одинаковое
  if len(V_cols) != len(W_cols):
      raise ValueError("Количество столбцов, начинающихся с 'V',  и 'W' должно быть одинаковым!")

  # Создание `dtype_dict`  динамически
  dtype_dict = {col: int for col in V_cols}

  # Преобразование типов  столбцов
  for i in V_cols + W_cols:
      tvr_with_types[i] = tvr_with_types[i].fillna(0).astype(int)

  # Применение `dtype_dict`
  tvr_with_types = tvr_with_types.astype(dtype_dict)

  # Расчет 'total_GO'
  tvr_with_types['total_GO'] = sum(abs(tvr_with_types[V_cols[i]]) * tvr_with_types[W_cols[i]] for i in range(len(V_cols)))

  tvr_with_types = tvr_with_types.merge(varyant_types, how = 'left', left_on='stroka', right_on='variant')
  tvr_with_types['PARAMS'] = tvr_with_types['C'].astype(str)+'_'+ tvr_with_types['N'].astype(str)+'_'+ tvr_with_types['P'].astype(str)+'_'+ tvr_with_types['E'].astype(str)+'_'+ tvr_with_types['FrId'].astype(str)+'_'+ tvr_with_types['MoveN'].astype(str)

  return tvr_with_types

def TVR_asis(TVR):
  

  

  # функция преобразования 3-5 в лист [3, 4, 5]

  def range_num_to_list(range_num):

    result = []
    for element in range_num.split(','):
      split_index = element.find('-')
      if split_index == -1:
        result.append(int(element))
      else:
        start = int(element[:split_index])
        end = int(element[split_index+1:])
        result.extend(list(range(start, end+1)))

    return result
  
  
  # основной модуль


  with open(TVR, encoding='utf8') as file:
      separator = file.readline()[0]  # Получаем элемент, который разделяет названия колонок
  with open(TVR, encoding='utf8') as file:
      column_names = file.readline().strip().split(separator)[2:]  # Получаем названия колонок начиная с третьего элемента
  data = []
  with open(TVR, encoding='utf8') as file:
      lines = file.readlines()[1:]  # Пропускаем первую строку с заголовками
      reader = csv.reader(lines, delimiter=' ')
      for row in reader:
          if len(row) > 2:
              row[2] = ' '.join(row[2:])  # Объединяем оставшиеся элементы в третий столбец
              data.append(row[:3])  # Добавляем только первые три элемента в список

  # Преобразуем список в DataFrame
  df = pd.DataFrame(data, columns=['stroka', 'stolbec', 'data'])

  # Изменяем тип данных на целочисленные
  df['stroka'] = df['stroka'].astype(int)
  df['stolbec'] = df['stolbec'].astype(int)

  # создаем паттерн, чтобы все стобцы были учтены
  stroka = [9999]*56
  stolbec = list(range(1,57))
  data = ['**']*56
  pattern_dic = {'stroka': stroka, 'stolbec': stolbec, 'data':data}
  pattern_df = pd.DataFrame(pattern_dic)

  df = pd.concat([df,pattern_df])
  df = df.replace(r'^\s*$', np.nan, regex=True)

  tvr_table = df.pivot(index='stroka', columns='stolbec', values='data') # разворачиваем таблицу
  tvr_table.columns = column_names
  tvr_table.reset_index(inplace = True)
  tvr_table



  return tvr_table


import pandas as pd
import numpy as np
import csv

def TVR_transform(TVR, varyant_types):
    # Загрузка данных по инструментам
    sec_data_file = '/content/drive/MyDrive/work_data/TVR/sec_tvr.csv'
    sec_data = pd.read_csv(sec_data_file, sep=',', index_col=False)
    sec_tvr = sec_data[['SECID', 'MINSTEP', 'STEPPRICE', 'PREVSETTLEPRICE', 'INITIALMARGIN', 'BUYSELLFEE', 'SCALPERFEE']]
    sec_tvr['full_price'] = sec_tvr['PREVSETTLEPRICE'] / sec_tvr['MINSTEP'] * sec_tvr['STEPPRICE']

    # Функция преобразования диапазонов
    def range_num_to_list(range_num):
        result = []
        for element in range_num.split(','):
            if '-' in element:
                start, end = map(int, element.split('-'))
                result.extend(range(start, end+1))
            else:
                result.append(int(element))
        return result

    # Обработка вариантов
    varyant_types = pd.read_csv(varyant_types, sep='=', header=0)
    varyant_types['variant'] = varyant_types['variant'].apply(range_num_to_list)
    varyant_types = varyant_types.explode('variant')

    # Чтение TVR файла
    with open(TVR, encoding='utf8') as file:
        separator = file.readline()[0]
        file.seek(0)
        column_names = file.readline().strip().split(separator)[2:]

    data = []
    with open(TVR, encoding='utf8') as file:
        reader = csv.reader(file, delimiter=' ')
        next(reader)  # Пропускаем заголовок
        for row in reader:
            if len(row) > 2:
                row[2] = ' '.join(row[2:])
                data.append(row[:3])

    # Создание основного DataFrame
    df = pd.DataFrame(data, columns=['stroka', 'stolbec', 'data'])
    df['stroka'] = df['stroka'].astype(int)
    df['stolbec'] = df['stolbec'].astype(int)

    # Добавление паттерна для всех столбцов
    pattern_df = pd.DataFrame({
        'stroka': [9999]*56,
        'stolbec': list(range(1, 57)),
        'data': ['**']*56
    })
    df = pd.concat([df, pattern_df]).replace(r'^\s*$', np.nan, regex=True)

    # Создание сводной таблицы
    tvr_table = df.pivot(index='stroka', columns='stolbec', values='data')
    tvr_table = tvr_table.reindex(columns=range(1, 57)).rename(columns=lambda x: column_names[x-1] if x-1 < len(column_names) else f'Col{x}')
    tvr_table.reset_index(inplace=True)

    # Фильтрация данных
    condition = (tvr_table['Start'] == 'True') & (tvr_table['Kill all'] != 'True')
    tvr_with_types = tvr_table[condition].copy()

    # Добавление столбца W 0
    v0_index = tvr_with_types.columns.get_loc('V 0')
    tvr_with_types.insert(v0_index + 1, 'W 0', np.nan)

    # Преобразование в длинный формат
    col_names = tvr_with_types.columns.tolist()
    tvr_melt = pd.melt(
        tvr_with_types.reset_index(),
        id_vars=['stroka'],
        value_vars=col_names[1:],
        var_name='stolbec',
        value_name='data'
    )

    # Объединение с данными инструментов
    merged = tvr_melt.merge(sec_tvr, how='left', left_on='data', right_on='SECID')
    merged[['INITIALMARGIN', 'full_price']] = merged[['INITIALMARGIN', 'full_price']].shift(2)

    # Обработка для GO и price
    def process_group(df_group, col_name):
        df_group['data'] = np.where(df_group[col_name] > 0, df_group[col_name], df_group['data'])
        pivot_df = df_group.pivot(index='stroka', columns='stolbec', values='data')
        pivot_df = pivot_df.reindex(columns=col_names[1:], fill_value=0)
        return pivot_df

    # Расчет для обоих столбцов
    go_pivot = process_group(merged.copy(), 'INITIALMARGIN')
    price_pivot = process_group(merged.copy(), 'full_price')

    # Объединение результатов
    final_df = tvr_table.merge(go_pivot, on='stroka', suffixes=('', '_GO'))
    final_df = final_df.merge(price_pivot, on='stroka', suffixes=('', '_Price'))

    # Расчет итоговых сумм
    v_cols = [c for c in final_df.columns if c.startswith('V')]
    w_cols = [c for c in final_df.columns if c.startswith('W')]

    final_df['total_GO'] = (final_df[v_cols].abs().values * final_df[w_cols].values).sum(axis=1)
    final_df['total_price'] = (final_df[[c for c in v_cols if '_Price' in c]].abs().values * 
                              final_df[[c for c in w_cols if '_Price' in c]].values).sum(axis=1)

    # Финализация структуры
    final_df = final_df.merge(varyant_types, left_on='stroka', right_on='variant')
    final_df['PARAMS'] = final_df[['C', 'N', 'P', 'E', 'FrId', 'MoveN']].astype(str).agg('_'.join, axis=1)
    
    return final_df[['stroka', 'total_GO', 'total_price', 'PARAMS'] + col_names]



def TVR_transform_CG(TVR, varyant_types):
    # ----------------------------------------------------------------------------
    # МОДУЛЬ ЗАГРУЗКИ ДАННЫХ ПО ФЬЮЧАМ/ОПЦИОНАМ
    sec_data_file = '/content/drive/MyDrive/work_data/TVR/sec_tvr.csv'
    sec_data = pd.read_csv(sec_data_file, sep=',', index_col=False)
    sec_tvr = sec_data[['SECID', 'MINSTEP', 'STEPPRICE', 'PREVSETTLEPRICE',
                        'INITIALMARGIN', 'BUYSELLFEE', 'SCALPERFEE']]
    # Добавим колонку 'full_price' (полная теоретическая цена)
    sec_tvr['full_price'] = sec_tvr['PREVSETTLEPRICE'] / sec_tvr['MINSTEP'] * sec_tvr['STEPPRICE']
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # МОДУЛЬ ПРЕОБРАЗОВАНИЯ "3-5" -> [3,4,5] для таблички вариантов
    def range_num_to_list(range_num):
        result = []
        for element in range_num.split(','):
            split_index = element.find('-')
            if split_index == -1:
                result.append(int(element))
            else:
                start = int(element[:split_index])
                end = int(element[split_index+1:])
                result.extend(list(range(start, end+1)))
        return result
    
    varyant_types = pd.read_csv(varyant_types, sep='=', header=0)
    varyant_types['variant'] = varyant_types['variant'].apply(range_num_to_list)
    varyant_types = varyant_types.explode('variant')
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # ЧТЕНИЕ ФАЙЛА TVR, КОТОРЫЙ ИМЕЕТ СВОЮ ОСОБУЮ РАЗМЕТКУ
    with open(TVR, encoding='utf8') as file:
        separator = file.readline()[0]  
    with open(TVR, encoding='utf8') as file:
        column_names = file.readline().strip().split(separator)[2:]  # названия колонок с 3-го элемента

    data = []
    with open(TVR, encoding='utf8') as file:
        lines = file.readlines()[1:]  # пропускаем строку с заголовками
        reader = csv.reader(lines, delimiter=' ')
        for row in reader:
            if len(row) > 2:
                # склеиваем всё, что после второго столбца, в один
                row[2] = ' '.join(row[2:])
                data.append(row[:3])

    # формируем DataFrame
    df = pd.DataFrame(data, columns=['stroka', 'stolbec', 'data'])
    df['stroka'] = df['stroka'].astype(int)
    df['stolbec'] = df['stolbec'].astype(int)

    # Добавим паттерн, чтобы учесть все столбцы при развороте
    stroka = [9999]*56
    stolbec = list(range(1, 57))
    data_pattern = ['**']*56
    pattern_dic = {'stroka': stroka, 'stolbec': stolbec, 'data': data_pattern}
    pattern_df = pd.DataFrame(pattern_dic)

    df = pd.concat([df, pattern_df])
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Делаем pivot, чтобы получить «широкую» таблицу (строка, столбцы)
    tvr_table = df.pivot(index='stroka', columns='stolbec', values='data')
    tvr_table.columns = column_names
    tvr_table.reset_index(inplace=True)
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # ФИЛЬТРЫ: берём только те, у кого 'Start' == 'True', но 'Kill all' != 'True'
    condition = (tvr_table['Start'] == 'True') & (tvr_table['Kill all'] != 'True')
    tvr_with_types = tvr_table[condition].copy()
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # ВСТАВЛЯЕМ СТОЛБЕЦ "W 0" РЯДОМ С "V 0" (как у вас было)
    stolbec_ryadom = 'V 0'
    index_stolbec = tvr_with_types.columns.get_loc(stolbec_ryadom)
    tvr_with_types.insert(index_stolbec+1, 'W 0', np.nan)

    # Запомним старые названия и сделаем нумерацию для melt
    col_names = tvr_with_types.columns.tolist()
    tvr_with_types = tvr_with_types.reset_index(drop=True)
    col_n = list(range(1, len(tvr_with_types.columns)+1))
    col_n.insert(0, 'stroka')  # в начало добавим 'stroka'
    tvr_with_types.insert(0, 'stroka', tvr_with_types.index)
    tvr_with_types.columns = col_n

    # Теперь "расплавим" (melt), чтобы потом можно было объединить с sec_tvr
    tvr_with_types = pd.melt(
        tvr_with_types,
        id_vars=['stroka'],
        var_name='stolbec',
        value_name='data'
    )
    tvr_with_types = tvr_with_types.sort_values(by=['stroka', 'stolbec'], ascending=True)
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # ПОДТЯГИВАЕМ ИЗ sec_tvr ОБЕ КОЛОНКИ: 'INITIALMARGIN' и 'full_price'
    tvr_with_types = tvr_with_types.merge(
        sec_tvr,
        how='left',
        left_on='data',
        right_on='SECID'
    )
    # сдвигаем оба поля на 2 строки вниз,
    # чтобы они вставали туда же, куда раньше вставляли по одному (col_count)
    tvr_with_types['INITIALMARGIN_shift'] = tvr_with_types['INITIALMARGIN'].shift(2)
    tvr_with_types['full_price_shift']    = tvr_with_types['full_price'].shift(2)

    # Теперь сформируем новые поля:
    #   data_margin - та строка, куда вписывается ГО, 
    #   data_price  - та строка, куда вписывается полная цена.
    # Логика та же: "если > 0, то берём это значение, иначе оставляем как было"
    tvr_with_types['data_margin'] = np.where(
        tvr_with_types['INITIALMARGIN_shift'] > 0,
        tvr_with_types['INITIALMARGIN_shift'],
        tvr_with_types['data']
    )
    tvr_with_types['data_price'] = np.where(
        tvr_with_types['full_price_shift'] > 0,
        tvr_with_types['full_price_shift'],
        tvr_with_types['data']
    )

    # После этого нам, чтобы вернуть структуру,
    # нужно «развернуть» pivot, но уже для нескольких столбцов сразу.
    # Воспользуемся pivot_table и укажем values = ['data_margin', 'data_price']
    tvr_with_types = tvr_with_types.pivot_table(
        index='stroka',
        columns='stolbec',
        values=['data_margin', 'data_price'],  # сразу 2 поля
        aggfunc='first'  # поскольку для каждой (stroka, stolbec) у нас одно значение
    )

    # Теперь у нас мультииндекс колонок: сверху ('data_margin' или 'data_price'),
    # снизу -- прежние названия столбцов (A, B, C, V 1, V 2, W 1, W 2, и т.д.)
    # Сформируем обычные "плоские" названия:
    tvr_with_types.columns = [
        f"{upper}_{lower}" for upper, lower in tvr_with_types.columns
    ]

    # Примерно у вас появятся колонки вида:
    #   data_margin_A, data_margin_B, ..., data_margin_V 1, data_margin_W 1, ...
    #   data_price_A, data_price_B, ..., data_price_V 1,  data_price_W 1,  ...
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # ДАЛЕЕ — ВОССТАНАВЛИВАЕМ СТРУКТУРУ, СОРТИРОВКУ, МЕРЖИМ С varyant_types И ПР.
    # (по аналогии с вашим кодом). 
    # Сразу же переименуем обратно "data_margin_<колонка>" в просто "<колонка>",
    # а "data_price_<колонка>" – в "p_<колонка>", чтобы отличать (или наоборот).
    # А можем оставить, как есть, – на ваше усмотрение.
    new_cols = {}
    for col in tvr_with_types.columns:
        # Если начинается с data_margin_, убираем это префикс
        if col.startswith("data_margin_"):
            new_cols[col] = col.replace("data_margin_", "")
        elif col.startswith("data_price_"):
            # для цены добавим букву p_ или как-то иначе обозначим
            base = col.replace("data_price_", "")
            new_cols[col] = "p_" + base  
    tvr_with_types.rename(columns=new_cols, inplace=True)

    # Теперь у нас колонки:
    #   A, B, C, V 1, V 2, ..., W 1, W 2, ...
    #   p_A, p_B, p_C, p_V 1, p_W 1, ...
    # где без префикса = ГО (или "то, что было в data_margin"),
    # а с префиксом p_ = полная цена (или "data_price").
    tvr_with_types.reset_index(inplace=True)

    # Чтобы продолжить, вам, скорее всего, нужно вернуть такие же текстовые
    # колонки, как и раньше (A,B,C... Mode, V 1, W 1...), а колонки p_A, p_B...
    # использовать только там, где надо числа. 
    # Но ниже код, показывающий идею, как посчитать total_GO и total_price.

    # Переведём в int те V- и W-колонки, которые числовые.
    # Идентифицируем их по шаблону 'V ' и 'W ':
    V_cols_margin = [c for c in tvr_with_types.columns if c.startswith('V ')]
    W_cols_margin = [c for c in tvr_with_types.columns if c.startswith('W ')]
    # То же самое для цены (префикс p_):
    V_cols_price = [c for c in tvr_with_types.columns if c.startswith('p_V ')]
    W_cols_price = [c for c in tvr_with_types.columns if c.startswith('p_W ')]
    
    # Чтобы не путаться, при необходимости всё, что НЕ похоже на столбцы V/W,
    # оставим в текстовом/объектном виде, а V/W сконвертируем в int/float:
    for col in (V_cols_margin + W_cols_margin + V_cols_price + W_cols_price):
        # заменим NaN на 0
        tvr_with_types[col] = pd.to_numeric(tvr_with_types[col], errors='coerce').fillna(0).astype(float)

    # Теперь считаем total_GO (на базе "обычных" V и W — это ГО)
    # total_GO = Σ |V_i| * W_i, i.e. по всем i
    # V_cols_margin[i] соответствует W_cols_margin[i] (по индексу в списке)
    # (если у вас не строго один к одному, надо аккуратнее подбирать соответствие)
    tvr_with_types['total_GO'] = 0
    for i in range(len(V_cols_margin)):
        tvr_with_types['total_GO'] += abs(tvr_with_types[V_cols_margin[i]]) * tvr_with_types[W_cols_margin[i]]

    # Аналогично total_price, но используем p_V ... и p_W ... (цена):
    tvr_with_types['total_price'] = 0
    for i in range(len(V_cols_price)):
        tvr_with_types['total_price'] += abs(tvr_with_types[V_cols_price[i]]) * tvr_with_types[W_cols_price[i]]

    # ----------------------------------------------------------------------------
    # ВОЗВРАТ К ТЕМ СТОЛБЦАМ, ЧТО ВАМ НУЖНЫ. ПРИМЕР: ДЕЛАЕМ МЕРЖ С varyant_types.
    # У вас в конце ещё идут различные сортировки, создание variant-колонки и т.д.
    # Примерно так:
    tvr_with_types = tvr_with_types.merge(
        varyant_types,
        how='left',
        left_on='stroka',
        right_on='variant'
    )

    # Добавим PARAMS (как у вас было)
    # (будьте внимательны, что колонки 'C','N','P','E','FrId','MoveN' теперь
    #   могут называться иначе, либо находиться в части "A, B...". 
    #   Скорее всего, нужно взять именно из "A" / "B" / "C" / ...? 
    #   Зависит от того, что было в исходном TVR.)
    #   Ниже просто пример:
    tvr_with_types['PARAMS'] = (tvr_with_types['C'].astype(str) + '_' + 
                                tvr_with_types['N'].astype(str) + '_' +
                                tvr_with_types['P'].astype(str) + '_' +
                                tvr_with_types['E'].astype(str) + '_' +
                                tvr_with_types['FrId'].astype(str) + '_' +
                                tvr_with_types['MoveN'].astype(str))

    # Итог: у нас есть tvr_with_types, где:
    #   -- обычные "текстовые" колонки (Mode, A, B, C...),
    #   -- дубли с префиксом p_ (p_A, p_B, p_C...),
    #   -- total_GO,
    #   -- total_price,
    #   -- PARAMS, и т. д.
    return tvr_with_types



