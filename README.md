# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

### some command line arguments
## train
```
python train.py ./flowers/ --save_dir ./checkpoint2.pth --arch vgg16 --epochs 5 --hidden_units 4096 --learning_rate 0.001 --dropout 0.5
```

## predict
```
python predict.py /home/workspace/ImageClassifier/flowersme/workspace/ImageClassifier/checkpoint2.pth 
```