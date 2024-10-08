def TVR_transform(TVR):
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
  return df