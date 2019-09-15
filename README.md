# Crowd-counter
This is a segmentation mask based AI (Mask R-CNN) for counting a crowd and silhouettes around the predictions
[[https://github.com/henrih13/Crowd-counter/blob/master/60confp2.PNG[alt=60conf2]]


-----------------TRAINING-----------------

Training (default weights, goes to "weight.h5"):
python human.py train

Training (other weights):
python human.py train --weights <path_to_the_weights.h5>


-----------------PREDICTION-----------------

Prediction (default image test directory, will use default weights):
python human.py splash

Prediction (custom directory)
python human.py splash --imagedir <imagedir_location>

Prediction (Custom weights)
python human.py splash --weights <path_to_the_weights.h5>


-----------------ERRORS---------------------

Possible errors if something goes wrong:
- You have to be where the code is located, i.e. cd to the location of "human.py"
- Python 3+ has to be used
- Pathing to weights or testimages could be incorrect


-----------------OTHER----------------------

Where are predictions being saved?
- They aren't
