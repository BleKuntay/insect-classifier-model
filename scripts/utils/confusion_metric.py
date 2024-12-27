import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Dapatkan prediksi dari model
predictions = model.predict(val_generator_encoded, steps=val_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)

# Pastikan jumlah true_labels sesuai dengan prediksi
true_labels = val_labels_encoded[:len(predicted_classes)]

# Hitung Confusion Matrix
cm = confusion_matrix(true_labels, predicted_classes)

# Visualisasi Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
