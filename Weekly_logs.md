# Weekly logs

These are logs that I will update every week in order to keep track of my work (and possibly so that other are also able to follow it).

## Table of contents
1. [Week 1 : 07/02](#week1)
2. [Week 2 : 14/02](#week2)
3. [Week 3 : 21/02](#week3)
   


## Week 1 : 07/02 <a name="week1"></a>

Start of the internship. Basically for this week I only read different papers on perceptual robust image hashing algorithms. I also tested some of the algorithms to verify some claims.

### Most interesting papers :

1. Efficient Cropping-Resistant Robust Image Hashing
2. Block Mean Value Based Image Perceptual Hashing
3. Histogram-Based Image Hashing Scheme Robust Against Geometric Deformations
4. Robust Hashing for Efficient Forensic Analysis of Image Sets (**Basically an improvement of paper 2**)
5. Forensics Investigations of Multimedia Data: A Review of the State-of-the-Art

### Some websites describing hash functions :

1. [average hash, phash](http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)
2. [dhash](http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)
3. [whash](https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5)


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

Once this was done, I forked the [imagehash library](https://github.com/JohannesBuchner/imagehash) to implement the histogram based hash algorithm (see paper 3 of [Week 1](#week1)). I also added some convenient methods in the base classes `ImageHash` and `ImageMultiHash` in order to easily tell if a given hash matches a hash inside a hash database.

Then, I created the scripts `create_groups.py` and `create_attacks.py` to easily divide the dataset into two subfolders (the control and experimental images) and create `N = 100` attacks in both groups. 
Finally the script `compare_algos.py` uses these groups to obtain the first results in the folder 
> test_performances/Results/General

These results are basically the ROC curve and the detailed metrics (accuracy, precision, recall, FPR) of all algorithms, over the performances on all the dataset aggregated. The mean running time of each method is also given.

------------------------------------------------------------

## Week 3 : 21/02 <a name="week3"></a>

The first thing this week was to change the text attack of the generator so that it is as close as possible to the *meme pattern*. I implemented it as white text with black edges. The font is now **Impact**, which is apparently the most used font for memes. I also increased the size of the font, so that it is now 40 for an image of width 512. 


I then created the script `compare_algos_details.py` in order to obtain the same ROC curves as the previous week but separately for each attack type and intensity, not aggregated over all images. This allowed to visualize that the algorithms are overall very robusts against *noise-like*  attacks (such as noise, filtering, modifying contrast, color,...) but struggle a lot against strong geometric deformation (large cropping, rotation or shearing). 

The script `mapping_to_images.py` was created to try to detect *pathological images* (images with which the algorithms struggle particularly). It gives for each image inside the experimental group (250 images making up the database) the number of time it was correctly identified (for the 100/250 images supposed to be identified) or incorrectly identified (for all 250 images of the experimental group). One image presented to the algorithms can have several match inside the database. 
From this I extracted heatmaps (or algorithms/BERs thresholds) showing the similarity between 20 least recognized images. In the same way, I looked at similarity proportion between 50 most wrongly identified images, only for high BERs, otherwise some algorithms have a false positive rate (FPR) of 0, making no sense to compare it to others.

Finally, the script `majority_vote.py` investigates if a majority vote between algorithms for classification gives better results that independant algorithms. Improves a bit but not too significantly. In the future, I believe that this can be improved by adapting the BERs thresholds for each algorithms inside the majority vote as all algorithms shows an overall different curve of FPR against BERs threshold.

Among other things, I also took some time to clean up the code for the generator, giving more meaningful variable names (also in some other scripts), and moved all plot generation in the file 
> test_performances/Create_plot.py 

I also began to read papers on Neural Hashing, and wrote those logs.

## Week 4 : 28/02 <a name="week4"></a>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-zv4m{border-color:#ffffff;text-align:left;vertical-align:top}
.tg .tg-8jgo{border-color:#ffffff;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-8jgo" rowspan="2">Algorithm</th>
    <th class="tg-8jgo" colspan="3">Recall</th>
    <th class="tg-8jgo" colspan="3">FPR</th>
  </tr>
  <tr>
    <th class="tg-8jgo">0.7</th>
    <th class="tg-8jgo">0.8</th>
    <th class="tg-8jgo">0.9</th>
    <th class="tg-8jgo">0.2</th>
    <th class="tg-8jgo">0.3</th>
    <th class="tg-8jgo">0.4</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-zv4m">Ahash</td>
    <td class="tg-8jgo">0.04</td>
    <td class="tg-8jgo">0.115</td>
    <td class="tg-8jgo">0.19</td>
    <td class="tg-8jgo">0.18</td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
  </tr>
  <tr>
    <td class="tg-zv4m">Phash</td>
    <td class="tg-8jgo">0.07</td>
    <td class="tg-8jgo">0.17</td>
    <td class="tg-8jgo">0.31</td>
    <td class="tg-8jgo">0.29</td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
  </tr>
  <tr>
    <td class="tg-zv4m">Dhash</td>
    <td class="tg-8jgo">0.05</td>
    <td class="tg-8jgo">0.15</td>
    <td class="tg-8jgo">0.27</td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
  </tr>
  <tr>
    <td class="tg-zv4m">Whash</td>
    <td class="tg-8jgo">0.04</td>
    <td class="tg-8jgo">0.12</td>
    <td class="tg-8jgo">0.18</td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
  </tr>
  <tr>
    <td class="tg-zv4m">CRhash</td>
    <td class="tg-8jgo">0.03</td>
    <td class="tg-8jgo">0.07</td>
    <td class="tg-8jgo">0.2</td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
    <td class="tg-8jgo"></td>
  </tr>
</tbody>
</table>

BER thresholds for constant recall/FPR for the different algos.