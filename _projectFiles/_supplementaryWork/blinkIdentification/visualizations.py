import matplotlib.pyplot as plt
import numpy as np
class plotting:
    def __init__(self, trainModel):
        self.trainingInstance = trainModel

    def model_performance(self):
        plt.figure(figsize=(12, 5))
        plt.suptitle(f'Model Performance')
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.trainingInstance.num_epochs + 1), self.trainingInstance.train_accuracies, label='Train Accuracy')
        plt.plot(range(1, self.trainingInstance.num_epochs + 1), self.trainingInstance.test_accuracies, label='Test Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.trainingInstance.num_epochs + 1), self.trainingInstance.train_f1_scores, label='Train F1 Score')
        plt.plot(range(1, self.trainingInstance.num_epochs + 1), self.trainingInstance.test_f1_scores, label='Test F1 Score')
        plt.title('F1 Score over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()

        plt.show()
        plt.savefig('model_performance.png')

    def classification_threshold(self):
        actual_labels, predicted_labels = self.trainingInstance.get_predictions()
        # Assuming outputs are from your model and testY_tensor is available
        probabilities = np.array(predicted_labels)

        thresholds = np.linspace(0, 1, num=25)
        curr_method_probs = probabilities[:, 0]

        TP_rates = [];
        FP_rates = [];
        TN_rates = [];
        FN_rates = []

        for threshold in thresholds:
            predicted = [0 if p >= threshold else 1 for p in curr_method_probs]

            TP = 0;
            FP = 0;
            TN = 0;
            FN = 0
            for i in range(len(predicted)):
                if predicted[i] == 0:
                    if actual_labels[i] == 0:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if actual_labels[i] == 1:
                        TN += 1
                    else:
                        FN += 1

            #
            TP_rates.append(TP / (len(predicted) - np.count_nonzero(actual_labels)))
            FP_rates.append(FP / np.count_nonzero(actual_labels))
            TN_rates.append(TN / np.count_nonzero(actual_labels))
            FN_rates.append(FN / (len(predicted) - np.count_nonzero(actual_labels)))

        plt.figure(figsize=(10, 7))
        plt.plot(thresholds, TP_rates, label='True Positives', color='green')
        plt.plot(thresholds, FP_rates, label='False Positives', color='red')
        plt.plot(thresholds, TN_rates, label='True Negatives', color='blue')
        plt.plot(thresholds, FN_rates, label='False Negatives', color='orange')
        plt.title(f'Classification Metrics as Function of Thresholding')
        plt.xlabel('Threshold Value')
        plt.ylabel('Percentage (%)')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('classification_threshold.png')