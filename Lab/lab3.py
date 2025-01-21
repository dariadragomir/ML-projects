import numpy as np
class KnnClasifier:
    def __init__(self, train_images, train_labels): 
        self.train_images = train_images 
        self.train_labels = train_labels
        
    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        if metric == 'l1':
            distances = np.sum(np.abs(self.train_images - test_image), axis=1)
        elif metric == 'l2':
            distances = np.sqrt(np.sum(((self.train_images - test_image) ** 2),axis=1))
        else:
            raise Exception("Metric not implemented")
        sorted_indices = distances.argsort()
        nearest_indices = sorted_indices[:num_neighbors]
        nearest_labels = self.train_labels[nearest_indices]
        
        return np.bincount(nearest_labels).argmax()
    
    def classify_images(self, test_images, num_neighbors=3, metric='l2'):
        predicted_labels = [self.classify_image(image,num_neighbors, metric) for image in test_images]
        
        return np.array(predicted_labels)
def accuracy_score(ground_truth_labels, predicted_labels):
    return np.mean(ground_truth_labels == predicted_labels)
train_images = np.loadtxt("data/train_images.txt")
train_labels = np.int32(np.loadtxt("data/train_labels.txt"))
test_images = np.loadtxt("data/test_images.txt")
test_labels = np.int32(np.loadtxt("data/test_labels.txt"))
print(train_images.shape)
print(test_images.shape)
'''(1000, 784)
(500, 784)'''
knn_clasifier = KnnClasifier(train_images, train_labels)
predicted_labels = knn_clasifier.classify_images(test_images, num_neighbors=3, metric="l2")
acc = accuracy_score(test_labels, predicted_labels)
print(acc)
#0.898
def get_accuracies(train_images, train_labels, test_images, test_labels, metric):
    knn_clasifier = KnnClasifier(train_images, train_labels)
    predicted_labels = knn_clasifier.classify_images(test_images, metric=metric)
    return accuracy_score(test_labels, predicted_labels)
acc_l2 = get_accuracies(train_images, train_labels, test_images, test_labels, metric = "l2")
acc_l1 = get_accuracies(train_images, train_labels, test_images, test_labels, metric = "l1")
 
