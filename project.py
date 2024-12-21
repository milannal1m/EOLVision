from get_data import get_train_val 
from build_model import load_model, build_model
from plot_model import plot_model
import matplotlib.pyplot as plt  

x_train, y_train, x_val, y_val = get_train_val()

#model,history = build_model(x_train, y_train, x_val, y_val, ['blue', 'fail', 'red', 'white'], 'first_ever_model')

model,history = load_model('first_ever_model')
model.summary()

train_score = model.evaluate(x_train, y_train,verbose = 0)
val_score = model.evaluate(x_val, y_val,verbose = 0)

print(f"Training accuracy: {train_score[1]:.2f}")
print(f"Validation accuracy: {val_score[1]:.2f}")

plot_model(history)


