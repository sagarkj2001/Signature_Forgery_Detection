# SIGNATURE FORYGERY DETECTION
 An Advanced level of Detecting the Forged Signature Images.

### Project Link:  [**Project.rar**](https://github.com/sagarkj2001/Signature_Forgery_Detection/blob/main/Project.rar)

 # INTRODUCTION
 Signature verification and forgery detection are the process of verifying signatures 
automatically and instantly to determine whether the signature is real or not. There are two main 
kinds of signature verification: static and dynamic. Static, or offline verification is the process of 
verifying a document signature after it has been made, while dynamic or online verification takes 
place as a person creates his/her signature on a digital tablet or a similar device. The signature in 
question is then compared to previous samples of that person's signature, which set up the 
database. In the case handwritten signature on a document, the computer needs the samples to be 
scanned for investigation, whereas a digital signature which is already stored in a data format can 
be used for signature verification. Handwritten signature is one of the most generally accepted 
personal attributes for verification with identity whether it may for banking or business. While 
this method uses **CNNs to learn the signatures**, the structure of our fully connected layer is not 
optimal and **GLCM is used to extract the texture features of the image.** In the model we will 
create two classes for each user real and forgery.

**NOTE- This project in repository does not contain the full code. Please contact me for Full code.**


### Project PPT [**Signature_Forgery_PPT.pptx**](https://github.com/sagarkj2001/Signature_Forgery_Detection/blob/main/Signature_Forgery_PPT.pptx)

### Project PPT [**Synopsis.pdf**](https://github.com/sagarkj2001/Signature_Forgery_Detection/blob/main/Synopsis.pdf)

# MODELING AND IMPLEMENTATION
**1. Creation of Dataset:** The images used for this project were collected from various 
internet sources and morphed using photo editing tools. These images were edited using 
Adobe Photoshop CC 2017 to create a dataset with more pairs of images- one original and 
its edited version. These images were used in further analysis using PYTHON. 
**2. Pre- Processing of the Images:** To make the details of the images stand out more, the 
query image was enhanced using histogram equalization. It is a necessary step because 
sometimes minute forgeries go undetected through the entire process. It is important that the 
machine gets most of the details in one go. Histogram equalization, as the name suggests, is 
a method, where the intensities are adjusted using the histogram of the image. This technique 
is used here for contrast enhancement. Another essential stage in the pre- processing of an 
image is the removal of noise. De-noising is again done so that the details of the image are 
sharper and are not missed while extracting the features of the image. In this paper, de-noising is done using the median filter in PYTHON. 
**3. Segmentation:** The image is segmented into 3 clustering by using k-means clustering. K-means clustering is a technique for quantizing vectors. This method divides the image into 
segments, each containing mutually exclave data. This is a common method when it comes 
to pattern recognition and machine learning. One of the segmented images is chosen based 
on the information contained in it. To determine this, the GLCM features of each segment 
are calculated and the segment with the highest mean is chosen. The GLCM of the segmented 
image are then compared with the original image using cross-validation, which gives another 
array, which is studied to determine whether an image is morphed or not, and function for 
the result is added based on that. 
**4. Features extraction:** Out of all the methods to analyses an image, extraction of GLCM 
features has proven to be efficient time and time again. The gray level covariance matrix is 
a tabulation that provides with statistical measures for texture analysis. This method 
considers the spatial relationship between the intensities of pixels in a gray-level image. In 
this paper, the GLCM features were calculated to study the differences in the original image 
and the digitally forged image. This gave 22 texture values (for each image) to work with, 
most of which were similar when it came to an image and its fraudulent counterpart. In 
practice, this would lead to redundancy and would also increase the time to run the algorithm. 
Also, the histogram of oriented gradient (HOG) features was calculated which gave another 
set of features for the original and the morphed image. The HOG values of the original and the morphed images were reasonably apart from each other, which meant that these values 
will be useful in differentiating the original document from the morphed one. 
**5. Dataset Split:** Divide the dataset into training, validation, and testing sets. The training 
set is used to train the model, the validation set is used to tune hyperparameters and monitor 
performance during training, and the testing set is used to evaluate the final model. 
**6. Model Architecture:** Design a CNN architecture specifically tailored for signature 
forgery detection. Initially, the classifier used for classification of dataset into two parts as 
original. A CNN is the most suitable classifier for two-class classification problems. It finds 
an equivalent hyper-plane which separates the whole data by a specific criterion that depends 
on the algorithm applied. It tries to find out a hyper-plane which is far from the closest 
samples on the other side of the hyperplane while still classifying samples. It gives the best 
generalization techniques because of the larger margin. So, Convolutional Neural Network 
(CNN) classifier was applied on the dataset. CNN networks are basically a system of 
interconnected neuron like layers. The interconnection of the network can be adjusted based 
on the number of available inputs and outputs making it ideal for a supervised learning. The 
CNN model was trained by providing 220 images. 
**7. Train the CNN using the training set:**  During training, optimize the 
model's weights and biases by minimizing a suitable loss function, such as cross-entropy 
loss, using an optimization algorithm Adam.
**8. Hyperparameter Tuning:** Fine-tune the hyperparameters of the model, such as learning 
rate, batch size, and regularization techniques (e.g., dropout), using the validation set. This 
process involves adjusting these parameters iteratively to achieve optimal performance.
**9. Model Evaluation:** Evaluate the trained model using the testing set. Measure 
performance metrics such as accuracy, precision, recall, and F1-score to assess the 
effectiveness of the model in detecting signature forgeries. Additionally, consider visualizing 
and analyzing misclassified samples to gain insights into potential areas for improvement. 
**10. Deployment and Monitoring:** Once satisfied with the model's performance, deploy it 
for real-world use. Implement an interface or API that allows users to input signature images 
and receive predictions. Continuously monitor the system's performance and collect user 
feedback to identify any potential issues or areas for improvement.
![Screenshot](https://github.com/sagarkj2001/Signature_Forgery_Detection/blob/main/Pictures/UI.png)

## Result
![Screenshot](https://github.com/sagarkj2001/Signature_Forgery_Detection/blob/main/Pictures/Result.png)


## Contact
### Email: sagarkj2001@hotmail.com
