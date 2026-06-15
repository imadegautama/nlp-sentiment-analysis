# Ringkasan Evaluasi EmoSense-ID

## Tugas: Sentimen

- **Accuracy**: 0.9315
- **Macro-F1**: 0.9311

```
              precision    recall  f1-score   support

    Negative     0.9125    0.9610    0.9361       564
    Positive     0.9547    0.8992    0.9261       516

    accuracy                         0.9315      1080
   macro avg     0.9336    0.9301    0.9311      1080
weighted avg     0.9327    0.9315    0.9313      1080
```

![Confusion Matrix Sentimen](confusion_matrix_sentimen.png)

## Tugas: Emosi

- **Accuracy**: 0.6361
- **Macro-F1**: 0.6086

```
              precision    recall  f1-score   support

       Anger     0.4158    0.5643    0.4788       140
        Fear     0.4899    0.5272    0.5079       184
       Happy     0.8449    0.7542    0.7970       354
        Love     0.6229    0.6728    0.6469       162
     Sadness     0.6716    0.5625    0.6122       240

    accuracy                         0.6361      1080
   macro avg     0.6090    0.6162    0.6086      1080
weighted avg     0.6570    0.6361    0.6429      1080
```

![Confusion Matrix Emosi](confusion_matrix_emosi.png)
