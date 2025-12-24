import pandas as pd

print("Загрузка данных...")
df = pd.read_csv('data/online_shoppers_intention.csv')

months = {
    'Feb': 2, 'Mar': 3, 'May': 5, 'June': 6, 'Jul': 7,
    'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
df['Month'] = df['Month'].map(months)

df['Revenue'] = df['Revenue'].astype(int)
df['Weekend'] = df['Weekend'].astype(int)

cat_cols = ['VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
df = pd.get_dummies(df, columns=cat_cols)

print("Сохранение данных в data/df_clean.csv")
df.to_csv('data/df_clean.csv', index=False)