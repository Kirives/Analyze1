import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

print("Чтение файла data/df_clean.csv...")
df = pd.read_csv('data/df_clean.csv')

X = df.drop('Revenue', axis=1)
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', HistGradientBoostingClassifier(
        random_state=100, 
        class_weight='balanced',
        learning_rate=0.05,
        max_depth=3,
        max_iter=100
    ))
])

print("Обучение модели...")
pipeline.fit(X_train, y_train)

print("Метрики на тестовой выборке:")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

print("Сохранение модели в models/model.pkl")
joblib.dump(pipeline, 'models/model.pkl')