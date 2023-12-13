import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings('ignore')


def make_confusion_matrix(model, X, y, threshold=0.5):
    # Отримання передбачень від моделі
    y_pred_probs = model.predict(X)
    # Конвертуємо ймовірності у бінарні класифікації на основі заданого порогу
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Створення матриці плутанини
    confusion_mtx = confusion_matrix(y, y_pred)

    # Візуалізація матриці плутанини
    plt.figure(dpi=80)
    sns.heatmap(confusion_mtx, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
                xticklabels=['fail', 'success'],
                yticklabels=['fail', 'success'])
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.show()


# Завантаження даних
df = pd.read_csv('model.csv')

# Визначення вхідних та вихідних змінних
X = df.drop(['success'], axis=1)  # Матриця ознак X
y = df['success']  # Цільова змінна y

# Розділення даних на тренувальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормалізація даних
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Створення моделі глибокого навчання
model = Sequential()

# Перший шар: Повнозв'язний шар
model.add(Dense(64, activation='relu', input_dim=X_train_sc.shape[1], kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
model.add(Dropout(0.3))

# Другий шар: Повнозв'язний шар
model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
model.add(Dropout(0.3))

# Вихідний шар
model.add(Dense(1, activation='sigmoid'))

# Оптимізатор зі зменшеною швидкістю навчання
optimizer = Adam(learning_rate=0.0005)

# Компіляція моделі
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Встановлення ранньої зупинки
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Навчання моделі
model.fit(X_train_sc, y_train, epochs=150, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

# Оцінка ефективності моделі
predictions = model.predict(X_test_sc)
predictions = (predictions > 0.5)

# Розрахунок метрик класифікації
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

# Виведення метрик
print(classification_report(y_test, predictions))
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')

# Розрахунок кривої ROC
fpr, tpr, thresholds = roc_curve(y_test, predictions)
print('FPR:', fpr)
print('TPR:', tpr)
roc_auc = auc(fpr, tpr)

# Матриця плутанини
make_confusion_matrix(model, X_test_sc, y_test, threshold=0.5)
