# Weekly logs

## Table of contents
1. [Week 1 : 07/02](#week1)
2. [Week 2 : 14/02](#week2)
   


## Week 1 : 07/02 <a name="week1"></a>

Start of the internship. Basically for this week I only read different papers on perceptual robust image hashing algorithms. I also tested some of the algorithms to verify some claims.

### Most interesting papers :

1. Efficient Cropping-Resistant Robust Image Hashing
2. Block Mean Value Based Image Perceptual Hashing
3. Histogram-Based Image Hashing Scheme Robust Against Geometric Deformations
4. Robust Hashing for Efficient Forensic Analysis of Image Sets (**Basically an improvement of paper 2**)
5. Forensics Investigations of Multimedia Data: A Review of the State-of-the-Art

I downloaded the BSDS500 image dataset mentioned in one of the paper and started the experiments on this.

------------------------------------------------------------


## Week 2 : 14/02 <a name="week2"></a>

The first goal for this week was to create a generator of attacks (or modification) of images. For this I created the module **generator**. This conveniently performs the attacks and save them to disk. 

The attacks of the generator are as follow :

- Noise :
    - Gaussian noise (default variances 0.01, 0.01, 0.05)
    - Speckle noise (default variances 0.01, 0.01, 0.05)
    - Salt & Pepper noise (default amounts 0.05, 0.1, 0.15)
- Filtering :
    - Gaussian filtering (default filter size 3x3, 5x5, 7x7)
    - Median filtering (default filter size 3x3, 5x5, 7x7)
- Jpg compression (default quality factors 10, 50, 90)
- Scaling (default ratios 0.4, 0.8, 1.2, 1.6)
- Cropping (default percentages 5, 10, 20, 40, 60)
- Rotation (default angles 5, 10, 20, 40, 60 degrees)
- Shearing (default angles 1, 2, 5, 10, 20 degrees)
- Contrast enhancement (default factors 0.6, 0.8, 1.2, 1.4)
- Color enhancement (default factors 0.6, 0.8, 1.2, 1.4)
- Brightness enhancement (default factors 0.6, 0.8, 1.2, 1.4)
- Sharpness enhancement (default factors 0.6, 0.8, 1.2, 1.4)
- Text adding in a *meme way* (default character length 10, 20, 30, 40, 50)

This makes a total of **58 attacks per images by default**.
