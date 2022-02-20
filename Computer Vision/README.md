## Computer vision

Here are the best computer vision projects I've made so far.

1. Cats vs Dogs: This is a classic computer vision classification problem. The objective is to create an AI that when presented with an image of a cat or a dog, can correctly label it as one or the other, for this I used Python + Keras + TensorFlow. I applied the transfer learning technique on the MobileNet model, which was pre-trained on ImageNet, with this I was able to train just the top layers with my data, saving a lot of time and still getting an accuracy of 98.3% on unseen data. The comments for this project are in English.

2. Malaria detection: The objective of this project was to create an AI that could read the microscope image of a cell and tell if it is infected with the plasmodium parasite, the parasite that causes malaria and is responsible for hundreds of thousands of deaths every year. I used Python + Keras + TensorFlow with a custom CNN created over the Keras Sequential class and trained on the malaria dataset from TensorFlow Datasets, it includes just over 27k images of infected and healthy cells with exaclty 50% for each class. At the end I got a 95.2% precision on unseen data. The comments for this project are in English.
