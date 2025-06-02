import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Data preparation: please fill in actual data
data = {
    'Team': ['CTBC Brothers', 'Uni-Lions', 'Wei Chuan Dragons', 'Fubon Guardians', 'Rakuten Monkeys'],
    'OBP': [0.345, 0.332, 0.321, 0.328, 0.338],
    'SLG': [0.420, 0.410, 0.395, 0.400, 0.415],
    'BB_K': [0.55, 0.45, 0.40, 0.42, 0.50],
    'ERA': [3.50, 4.10, 4.50, 4.30, 3.90],        # 新增：投手防禦率
    'WHIP': [1.25, 1.32, 1.38, 1.35, 1.28],       # 新增：投手WHIP
    'WinRate': [0.580, 0.510, 0.460, 0.470, 0.500]
}

# 2. Create DataFrame
df = pd.DataFrame(data)

# 3. Define features and target
X = df[['OBP', 'SLG', 'BB_K', 'ERA', 'WHIP']]
y = df['WinRate']

# 4. Train the model
model = LinearRegression()
model.fit(X, y)

# 5. Display model parameters
print("\n--- Model Parameters ---")
print(f"Intercept: {model.intercept_:.4f}")
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")

# 6. Predict and append to DataFrame
predicted = model.predict(X)
df['PredictedWinRate'] = predicted

print("\n--- Actual vs Predicted Win Rates ---")
print(df)

# 7. Plot actual vs predicted win rates
plt.figure(figsize=(8, 6))
plt.scatter(df['WinRate'], df['PredictedWinRate'], color='blue', label='Predicted Points')

# 正確的 for 迴圈和 plt.text
for i in range(len(df)):
    plt.text(df['WinRate'][i] + 0.002, df['PredictedWinRate'][i], df['Team'][i], fontsize=10)

plt.plot([0.4, 0.65], [0.4, 0.65], '--', color='gray', label='Perfect Prediction Line')
plt.xlabel('Actual Win Rate')
plt.ylabel('Predicted Win Rate')
plt.title('Actual vs Predicted Win Rate (Team Labels)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

