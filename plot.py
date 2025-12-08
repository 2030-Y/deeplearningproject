import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Model.csv")  


#  Accuracy 

plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["accuracy"], marker='o', label="Train Accuracy")
plt.plot(df["epoch"], df["val_accuracy"], marker='o', label="Validation Accuracy")
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=300)
plt.show()


# Loss 

plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["loss"], marker='o', label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], marker='o', label="Validation Loss")
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=300)
plt.show()