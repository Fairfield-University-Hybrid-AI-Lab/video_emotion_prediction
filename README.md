# Video Emotion Prediction


The goal of this project is to develop a video emotion recognition system that is capable of online emotion prediction with constrained faces.
This could be applied in situations where regular emotion recognition is not possible due to various reasons. The target scenario that we are focused on is situations when a 
person is using a virtual reality headset or brain activity sensors.

This repository contains:

1. [Bounding box and annotation overlay code](danushka/boundingbox.py) for testing the existing video dataset.
2. [Extract faces from videos](danushka/extractfaces.py)
3. [Calculate class edges for affectnet dataset (csv file) using quantiles](danushka/quantilescript.py)
4. [Calculate class edges for emotion in the wild dataset using annotations](danushka/annotations_quantilescript.py)



## Table of Contents

- [Background](#background)
- [Install](#install)
- [License](#license)

## Background

In this project, the team will use publicly available facial expression datasets to create a large-scale dataset for emotion research using brain activity sensors. Currently available emotion datasets like affectnet [1] contain unconstrained images, however this is not the case that researchers encounter when combining brain activity sensors with video facial expression capturing. These brain activity sensors usually cover the top portion of the head (figure 2). This means that general emotion datasets like affectnet are not that useful for such research. Therefore, we propose the development of modified affectnet dataset for brain activity research.


## Install


## License

[MIT](LICENSE) © Danushka Bandara et al.
