# EMOJIANTOR 
<h4>
This program converts your facial expression into corresponding facial emoji. The model is trained on 450 images of customized faces for each emoji.Right now it supports the following 5 emojis :- <br></h4>
<h2>
1 - 🙂 <br>
2 - 🤫 <br>
3 - 😉 <br>
4 - 😀 <br>
5 - 😑 <br></h2>

<h2>Demo</h2>
<img src="https://github.com/pranavmicro7/Emojinator/blob/master/outputs/gif.gif"><br>

<h4><b>NOTE:-</b> The model is trained on the dataset on my face. So it is suggested to recreate dataset by running face_recorder.py.<br></h4><br>

<h2> Installations/Requirements</h2>
<h4>
1-Keras </br>
2-OpenCV </br>
3-Pickle </br>
4-Dlib </br>
5-imutils </br> 
6-shape_predictor_68_face_landmarks.dat
</h4>
<h2> Steps </h2>
<h4>
1- Firstly run face_recorder.py. This will save the customized faces for each emoji into their corresponding labelled folder.<br>
2- Then run dataset_creator.py to create train and test image dataset.<br>
3- Then run training_model.py to create prediction model.<br>
4- Finally run run.py to convert your real time facial expresiion into emoji.<br>
</h4>
