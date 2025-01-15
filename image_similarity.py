 # -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import cv2 as cv
import os
import random
import pickle


app = Flask(__name__)

upload_folder = os.path.join('uploads')

save_img = os.path.join('static')

app.config['UPLOAD'] = upload_folder

app.config['SAVE_IMAGE'] = save_img


# Load saved model from pickle file
model = pickle.load(open("model_saved.pkl", "rb"))

@app.route('/')
def upload():
    return render_template('upload.html')

@app.route('/graph')
def graph():
	return render_template('performance.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Check if a file was uploaded in the request
    if request.method == 'POST':
        
        # Load the image file from the request
        img_upl = request.files['image']
        
        # Save the uploaded image to a file in a specific directory
        img1_path = os.path.join(app.config['SAVE_IMAGE'], secure_filename(img_upl.filename))
        img_upl.save(img1_path)
        # Load the saved image using cv.imread()
        img1 = cv.imread(img1_path)
        
        
        scalled_raw_img = cv.resize(img1, (64, 64))
        X_test = np.array([scalled_raw_img])
        
        # Obtain embeddings of test image(s)
        embedder = FaceNet()
        X_test_embeddings = embedder.embeddings(X_test)
        
        # Predict class of test image(s) using loaded model and embeddings
        y_pred = model.predict(X_test_embeddings)
        
        # Print predicted class
        classes= ['chris_evans', 'chris_hemsworth', 'mark_ruffalo', 'robert_downey_jr', 'scarlett_johansson']
        predicted_class = classes[y_pred[0]]
        
        
        words = predicted_class.split('_')
        capitalized_words = [word.title() for word in words]
        result = ' '.join(capitalized_words)
        # print("Predicted class: ", result)
        
        img2 = 'img/' + predicted_class
        
        #-------------------
        
        # Get a list of all the image files in the 'path/to/c' folder
        image_files = [f for f in os.listdir(img2) if f.endswith('.png')]
        
        # Filter out the 'image1.png' file from the list
        image_files = [f for f in image_files if f != img1]
        
        # Select a random image file from the remaining list
        random_image = random.choice(image_files)
        
        # Construct the path to the selected image file
        path_to_random_image = os.path.join(img2, random_image)
        
        #-------------------
        
        # Set the input size of the images
        input_size = (224, 224)
        
        # Load the two images to compare
        image1 = img1
        image1 = cv.resize(image1, input_size)
        image2 = cv.imread(path_to_random_image)
        # image2 = cv.imread("F:/Flask/2022 - 2023/PROJECTS/VTPML06/img/chris_evans/chris_evans1.png")
        image2 = cv.resize(image2, input_size)
        
        # Convert the images to grayscale
        gray1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
        gray2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
        
        
        # Flatten the image arrays
        image1_features = gray1.reshape(1, -1)
        image2_features = gray2.reshape(1, -1)
        
        # Compute the cosine similarity between the feature vectors
        similarity_score = cosine_similarity(image1_features, image2_features)[0][0]
        
        if similarity_score>0.7:
            print("Similarity score:", round(similarity_score*100,0),"%")
            print("Person Matched")
        else:
            print("Similarity score:", round(similarity_score*100,0),"%")
            print("Person Unmatched")
            
        #--------------------
        
        MIN_MATCH_COUNT = 10
        
        
        # Initialize the SIFT detector
        sift = cv.SIFT_create()
        
        # Find the keypoints and descriptors for the two images
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # Initialize the FLANN matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        
        # Match the keypoints using FLANN
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Filter the matches using the Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.9  * n.distance:
                good.append(m)
        
        # Check if the number of good matches is above the threshold
        if len(good) >= MIN_MATCH_COUNT:
            # Estimate the homography between the two images using RANSAC
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        
            # Calculate the homography between the two images
            h, w = image1.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)
        
            # Draw the matches and the homography
            draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                                singlePointColor = None,
                                matchesMask = None, # draw all matches
                                flags = 2)
            img3 = cv.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)
            img3 = cv.polylines(img3, [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
            # save the output image to disk
            cv.imwrite('static/output.jpg', img3)
            image_path = os.path.join('static/output.jpg')
            print("image_path")
            
        else:
            print("Not enough matches found")
        
        
        return render_template('result.html', result=result,img = image_path, score=round(similarity_score*100,0))





if __name__ == '__main__':
	app.run()