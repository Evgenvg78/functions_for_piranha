import pandas as pd
import numpy as np
import glob
from moexalgo import Market, Ticker


def stat_df (dir,
             tvr_with_types,
             default_comis = 0.4,
             INSTRUMENT_TYPE = '/content/drive/MyDrive/work_data/TVR/INSTRUMENT_TYPE.csv'
             ):
  ''' Эта функция преобразует данные из папки my deals в датафрейм.

  Args:
    dir (str): путь к папке с файлами my deals.
    default_comis: комиссия брокера за сделку (в одну сторону) за контракт.
    INSTRUMENT_TYPE: ссылка на файл с указанием типов комиссия биржи (индекс, фонд, валюта, товар).
    
    tvr_with_types: ссылка на TVR (обработанный файл TVR).
  Returns:
    df: датафрейм с обработанными данными, где каждая строка это сделка - вход, выход, доходность и т.д.


  '''
  dir  = glob.glob(dir)

  

  # загрузка дынных с биржи

  # forts_tickers = Market('FO').tickers()
  # sec_data_list = []
  # for i in range(len(forts_tickers)):
  #   # print(i)
  #   name = forts_tickers.iloc[i, 0]
  #   sec = Ticker(name).info()
  #   sec = sec.T
  #   sec_data_list.append(sec)

  # sec_data = pd.concat(sec_data_list, axis=0).reset_index(drop=True)
  sec_data_file = '/content/drive/MyDrive/work_data/TVR/sec_tvr.csv'
  sec_data = pd.read_csv(sec_data_file, sep=',', index_col=False)             
  

  SEC_i = sec_data[['SECID', 'MINSTEP', 'STEPPRICE', 'PREVSETTLEPRICE', 'INITIALMARGIN', 'BUYSELLFEE', 'SCALPERFEE']]
  SEC_i.loc[SEC_i['SECID'].str.len()==4, 'SECID'] = SEC_i['SECID'].str[:2]
  SEC_i = SEC_i.groupby('SECID').mean().reset_index()
  SEC_i.rename(columns={'SECID': 'CODE'}, inplace=True)


  files = []
  for deal in dir:
    add = pd.read_csv(deal, sep=';', header=None)
    files.append(add)
  df = pd.concat(files, ignore_index=True)

  # изменяем структуру данных

  col_names = ['var', 'in_out', 'date_time', 'sec', '4','amount', 'l_sh', 'price', '8', 'order']
  df.columns = col_names
  df.drop(['4', '8'], axis = 1, inplace = True) # удаляем ненужные колонки
  df.loc[df['sec'].str.len() == 4, 'sec'] = df['sec'].str[:2] #убираем последние 2 символа в названии фьючерса
  df['date_time'] = pd.to_datetime(df['date_time'], format = '%d.%m.%Y %H:%M:%S') #
  df['date'] = df['date_time'].dt.date  # выделение даты в отдельный столбец
  df['price'] = df['price'].str.replace(',', '.').astype('float').round(2) # приведение цены к нормальному формату
  df = df.sort_values(by=['var', 'date_time'], ascending = [True, True]) # сортировка по роботу и по дате-времени

  df['rab_vs'] = np.where(df['in_out'] == 'V', 'vs', 'rab')

  # пока датасет чистый (без группировок) нужно определить in out для V.
  def in_out_V(group):
      group['full_i_o'] = group['in_out'].where((group['in_out'] == 'I') |(group['in_out'] == 'O'), np.nan)
      group['full_i_o'] = group['full_i_o'].ffill()
      return group
  df = df.groupby(['var']).apply(in_out_V).reset_index(drop=True)



  # убираем первые строки с символом "O" в каждом варианте с каждым инструментом
  def count_first_out(ser):
      count = 0
      for i in ser['full_i_o']:
          if i == 'O':
              count += 1
          else:
              break
      return ser.iloc[count:]
  df = df.groupby(['var']).apply(count_first_out).reset_index(drop=True)


  def count_lact_in(ser):
      count = 0
      for i in ser['full_i_o'].iloc[::-1]:
          if i == 'I':
              count += 1
          else:
              break


      if count > 0:
          return ser.iloc[:-count]
      else:
          return ser  # Используем отрицательное значение для среза для удаления последних count строк
  df = df.groupby(['var']).apply(count_lact_in).reset_index(drop=True)


  df['ind_i_o'] = df.groupby(['var'])['full_i_o'].transform (lambda x: (x != x.shift(1)).cumsum())
  df = df.sort_values(by=['var', 'ind_i_o', 'in_out'], ascending = [True, True, True]).reset_index(drop=True)


  df = df.groupby(['var', 'ind_i_o', 'sec', 'in_out']).agg({'var': 'last',
                                                    'in_out': 'last',
                                                    'date_time': 'last',
                                                    'sec': 'last',
                                                    'amount': 'sum',
                                                    'l_sh': 'last',
                                                    'price': 'mean',
                                                    'order': 'last',
                                                    'date': 'last',
                                                    'full_i_o': 'last',
                                                    'ind_i_o': 'last',
                                                    'rab_vs': 'last'}).reset_index(drop=True)
  df['price'] = df['price'].round(2)
  df = df.sort_values(by=['var', 'rab_vs', 'sec', 'ind_i_o'])

  # модуль расчета комиссии
  df['ord_ind'] = np.where(df['full_i_o'] == 'I', df['var'].astype(str) + '_' + df['date_time'].astype(str) + '_' + df['sec'].astype(str)+'_'+df['rab_vs'], None)
  df['ord_ind'] = df.groupby('var')['ord_ind'].transform(lambda x: x.ffill())
  df_I = df[df['full_i_o']=='I']
  df_O = df[df['full_i_o']=='O']
  full_df = pd.merge(df_I, df_O, on = 'ord_ind', suffixes = ('_i', '_o'))


  full_df['sec_err'] = np.where(full_df['sec_i'] == full_df['sec_o'], 'ok', 'err')
  full_df = full_df.loc[full_df['sec_err'] == 'ok']
  full_df['default_comis'] = 0.4

  full_df = full_df.merge(SEC_i, how='left', left_on = 'sec_i', right_on = 'CODE')

  INSTRUMENT_TYPES = pd.read_csv(INSTRUMENT_TYPE, sep=';', header=0)
  INSTRUMENT_TYPES['comis_koef']=INSTRUMENT_TYPES['comis_koef'].str.replace(',', '.').astype('float')
  full_df = full_df.merge(INSTRUMENT_TYPES, how='left', left_on = 'sec_i', right_on = 'CODE_in')
  df = full_df

  # окончательный расчет комиссии и доходности

  df['rub_price'] = df['price_o']/df['MINSTEP']*df['STEPPRICE'] # считаем стоимость инструмента в рублях
  df['count_comis'] = df['rub_price']/100 * df['comis_koef'] # вычисляем
  df['moex_comiss'] = np.where(df['count_comis'].isnull() | (df['count_comis'] == ''), df['BUYSELLFEE'], df['count_comis']) # берем либо расчетную комиссию либо последнюю
  df['final_comiss'] = np.where(df['rab_vs_i'] == 'rab', df['default_comis']*2, (df['default_comis']+df['moex_comiss'])*2*df['amount_o']) # считаем финальную комиссию (берем на вход и выход)
  df['profit'] = np.where(df['l_sh_i']>0,(df['price_o'] - df['price_i']) * df['amount_i'],(df['price_i'] - df['price_o']) * df['amount_i'])
  df['profit_rub'] = (df['profit']/df['MINSTEP']*df['STEPPRICE'] - df['final_comiss'])#.round(2)
  df['profit_rub_One'] = df['profit_rub']/df['amount_i']

  full_df = df


  # преобразуем файл var_group_ulia.txt с описанием скриптов в формат таблицы

  # функция преобразования 3-5 в лист [3, 4, 5]






  # окончательная разбивка на варианты, типы скриптов и др.
  full_df = full_df.merge(tvr_with_types, how='left', left_on='var_i', right_on='stroka') # варианты
  # full_df = full_df.merge(varyant_types, how='left', left_on='var_i', right_on='variant') # типы скриптов
  # получаем варианты, если в таблице TVR их не было
  full_df['var_i_sec'] = np.where(full_df['l_sh_i']==1,
                                  (full_df['var_i'].astype(str) + '-' + (full_df['var_i'].astype(int) + 1).astype(str)),
                                  ((full_df['var_i'].astype(int) - 1).astype(str) + '-' + full_df['var_i'].astype(str)))
  full_df['total_GO'] = full_df['amount_i']*full_df['INITIALMARGIN']
  full_df['variant_final'] = np.where(full_df['variant_x'].isnull(), full_df['var_i_sec'], full_df['variant_x'])
  full_df['total_GO'] = pd.to_numeric(full_df['total_GO']).round(2)
  full_df['script_type'] = full_df['script_type'].fillna('other')

  full_df = full_df[['var_i',
                    'date_time_i',
                    'sec_i',
                    'amount_i',
                    'l_sh_i',
                    'date_time_o',
                    'final_comiss',
                    'profit',
                    'profit_rub',
                    'script_type',
                    'variant_final',
                    'total_GO'
                    ]]
  return full_df





