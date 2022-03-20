![plot](https://user-images.githubusercontent.com/10033026/159179539-83296cd2-8484-4d90-b354-0f6d672d82fe.png)

# MIT-Capstone-Project
A Deep Learning project I am working on as a part of my Applied Data Science Program with MIT

## **Malaria Detection**

Malaria is a contagious disease caused by Plasmodium parasites that are transmitted to humans through the bites of infected female Anopheles mosquitoes. The parasites enter the blood and begin damaging red blood cells (RBCs) that carry oxygen, which can result in respiratory distress and other complications. The lethal parasites can stay alive for more than a year in a person’s body without showing any symptoms. Therefore, late treatment can cause complications and could even be fatal. Almost 50% of the world’s population is in danger from malaria. There were more than 229 million malaria cases and 400,000 malaria-related deaths reported over the world in 2019. Children under 5 years of age are the most vulnerable population group affected by malaria; in 2019 they accounted for 67% of all malaria deaths worldwide. Traditional diagnosis of malaria in the laboratory requires careful inspection by an experienced professional to discriminate between healthy and infected red blood cells. It is a tedious, time-consuming process, and the diagnostic accuracy (which heavily depends on human expertise) can be adversely impacted by inter-observer variability. An automated system can help with the early and accurate detection of malaria. Applications of automated classification techniques using Machine Learning (ML) and Artificial Intelligence (AI) have consistently shown higher accuracy than manual classification. It would therefore be highly beneficial to propose a method that performs malaria detection using Deep Learning Algorithms.

## **Objective**

#### Build an efficient computer vision model to detect malaria. The model should identify whether the image of a red blood cell is that of one infected with malaria or not, and classify the same as parasitized or uninfected, respectively.

- The parasitized cells contain the Plasmodium parasite
- The uninfected cells are free of the Plasmodium parasites but could contain other impurities

## **Steps to Solve**

1. Importing Libraries.
2. Loading the data.
3. Data preprocessing.
4. Data augmentation.
5. Ploting images and its labels to understand how does an infected cell and uninfected cell looks like.
6. Spliting data in Train , Evaluation and Test set.
7. Creating a Convolution Neural Network function.
8. Wrapping it with Tensorflow Estimator function.
9. Training the data on Train data.
10. Evaluating on evaluation data.
11. Predicting on Test data
12. Ploting the predicted image and its respective True value and predicted value.
