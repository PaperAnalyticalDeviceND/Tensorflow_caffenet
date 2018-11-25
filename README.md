# Tensorflow_caffenet

This code represents the Tensorflow version of Caffenet (the NN used in the PADs paper).
Included in the repo. is a set of images (generated using neural_augment_distort.py) used for the initial training session to validate the code. The parameters used to create the training set were,
```
python neural_augment_distort.py -s "Amoxicillin rerun,Acetaminophen,Ciprofloxacin,Ceftriaxone,Metformin,Ampicillin,Azithromycin,Cefuroxime Axetil,Levofloxacin" -t "12LanePADKenya2015" -e "AJ" -c 'General' -n 100 -f images -l label.txt
```
This script uses PAD cards from test "12LanePADKenya2015" from category "General" to train on drugs Amoxicillin rerun, Acetaminophen, Ciprofloxacin, Ceftriaxone, Metformin, Ampicillin, Azithromycin, Cefuroxime Axetil and Levofloxacin. 

It excludes lanes "AJ" and creates 100 distorted/stretched augmented images for training.

To train the network we use the Jupyter notebook "pad_caffenet.ipynb" and we can predict from a pad image using "predict.py". to use predict we need to include the image to be identified which should be a rectified image,
```
python3 predict.py -i /var/www/html/joomla/images/padimages/processed/Acetaminophen-12LanePADKenya2015-1-58861.processed.png
```
We can also set a network configuration file using the '-n' parameter,
```
python3 predict.py -n tensor_100_9.nnet -i /var/www/html/joomla/images/padimages/processed/Acetaminophen-12LanePADKenya2015-1-58861.processed.png
```
tensor_100_9.nnet contains the following,
```
DRUGS,Amoxicillin rerun,Acetaminophen,Ciprofloxacin,Ceftriaxone,Metformin,Ampicillin,Azithromycin,Cefuroxime Axetil,Levofloxacin
LANES,AJ
WEIGHTS,tmp/caffenet_pad_1.ckpt
TYPE,tensorflow
DESCRIPTION,9 drug NN generated using color variation of +/-0.5 and +/-75px height distortion.
TEST,12LanePADKenya2015
```
this allows the user to define the network training parameters but not the architecture.
