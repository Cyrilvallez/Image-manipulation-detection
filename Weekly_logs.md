# Weekly logs

These are logs that I will update every week in order to keep track of my work (and possibly so that other are also able to follow it).

## Table of contents
1. [Week 1 : 07/02](#week1)
2. [Week 2 : 14/02](#week2)
3. [Week 3 : 21/02](#week3)
4. [Week 4 : 28/02](#week4)
5. [Week 5 : 07/03](#week5)
6. [Week 6 : 14/03](#week6)
7. [Week 7 : 21/03](#week6)

   


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

I began this week with a lot a paper reading. First on GANs and Inception Score/Frechet Inception Distance (IS/FID), then on modern computer vision architectures (Inception, ResNet,...), and finally on self-supervised methods for learning a consistant image representation in the Euclidean space. 

The most interesting papers on self-supervised methods were :

1. A Simple Framework for Contrastive Learning of Visual Representations
2. FaceNet: A Unified Embedding for Face Recognition and Clustering
3. Revisiting Self-Supervised Visual Representation Learning
4. Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering

Paper 3. provides a very nice overview of what is efficient or not, and would be worth re-reading in the future before trying to train/fine-tune networks.

This took between 2 and 3 days to understand and be comfortable with all this material. In the end of the week, I explored the idea of the previous week of mapping image detection to images to try to detect pathological images. For this, the following table summarizes the BER thresholds for a constant recall and FPT between algorithms. It was possible to obtain (almost) equal recall by modyfiying the thresholds for each algorithm, however, it was not the case for FPR, as for example for Phash, a change in BER threshold corresponding from just one more bit allowed to flip would result in FPR going from about 0.2 to 0.8. Thus, it is possible to fairly compare images which are less detected in images that are supposed to be detected, but not images that are most detected in images not supposed to be detected.

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


After this was done, I started to experiment with neural hashing. First, I took the pre-trained Inception v3 network from PyTorch, removed the last layer mapping features to classes in the training set (ImageNet), and multiplied those features with a random matrix to obtain a binary hash of a given length. All images are pre-processed in the same fashion as they were for training. The performance was not as satisfactory as I hoped so, as it was quite clearly worse than basic hash algorithms.

I also took some time to setup a clean environment for me to work on the GPU cluster, and to understand how the setup was going to work.

The goal for next week is to use a pre-trained SimCLR model from paper 1., which I think would be better as it was train in self-supervised fashion, compared to Inception v3 which was trained for classification and whose features might be biased for this task instead of offering a clear representation of the image.


## Week 5 : 07/03 <a name="week5"></a>

In the spirit of what I started last week, I first looked at the SimCLR approach to perform neural hashing. The authors provided their code and pre-trained model in tensorflow, but by chance someone coded a conversion to pytorch. I used this, and took the SimCLR v1 ResNet 2x model to use as a model for neural hashing. As before, the images are pre-processed in the same way as they were for training. This model performed better than the Inception v3 model trained on ImageNet for hashing, but not much. 

My first thought was then towards the hashing process : transforming the features to binary hash using a random matrix may not be satisfactory, in the sense that we may loose too much information. Indeed, multiplying with a random matrix and assigning 0 or 1 depending on the sign of the output corresponds to just looking at the side of which the points are from random hyperplanes (represented by random normal vectors, all of which aggregated corresponds to the random matrix). To verify this hypothesis, I had to compare the performances when using the raw features for matching the images. But for this, one needs a coherent distance for thresholding and decisin-making (this image is the same as this one or not), which in the case of a binary hash is the difference in bits and BER threshold. As the Euclidean distance is not bounded and there is no way to know the maximum a priori (before hashing all images), I used the cosine distance, which can be very conveniently normalized between 0 and 1, allowing to directly compare to the hashing process using the same thresholds as the BER thresholds (which are also comprised between 0 and 1). Moreover SimCLR was trained on cosine distance. This approach provided a large increase in performance, comforting the idea that multiplying with a matrix looses information, or is at least not an optimal way to transform the features into binary hashes. 

The remaining part of the week was mostly used to create a framework allowing me to easily work with neural hashing in mini-batches instead of image after image (as with standard hashing) for performance, as well as allowing me to use in the same way neural hashing and classical hashing scheme, for easy comparison between all methods. Different other improvements were added in the code, for example in the generation of figures. The goal of this work is to be able to extensively test the hashing schemes next week in a convenient and easy manner, and on a much bigger scale (bigger datasets).


## Week 6 : 14/03 <a name="week6"></a>

This week was mostly used to finish the hashing framework, despite what I hoped. It actually took way longer than expected. I experimented quite a bit with datasets representation for fast batch picking. Mental note : I was able to go much faster in the process and abstraction once I decided to design (and actually code) what would be the "final function" (function reusing all the abstraction and calling everything else to concretely perform the hashing on all images) before continuing to design the rest of the library. This seems like a pretty reasonable and efficient approach rather than experiment with something, then change it because it does not match with another thing etc...

I also implemented the Jensen-Shannon distance in the library, as Andrei suggested the week before. I created a whole bunch of helper function to save and load the result of experiment, and had to rewrite the plotting logic for use with the new experiment digests obtained though the new hashing library. 

After that, I was finally able to perform the first benchmark of my new framework on the cluster. The speed results were quite good, with only between and 1 and 2 min for both neural models with cosine distance. Interestingly, it seems much longer with Jensen-Shannon distance (up to 7 min 30 s for SimCLR, compared to 1 min 30 with cosine) which is a very strong overhead for just a change in the distance. I will need to investigate this fact next week.
But the most interesting result of the week is that SimCLR combined with Jensen-Shannon seems to have a false positive rate (FPR) of 0, up to a threshold as high as 0.4, where most other algorithms are always predicting a match (FPR of 1) for such a threshold. Moreover, the curves for precision and recall are increasing along with the threshold. Thus by augmenting the threshold, it should be possible to find the point where the FPR starts to increase and maybe get very nice performances with very low FPR. 

I compared the results from this experiment to results I already had for some of the algorithms, and they are exactly identical, indicating that the whole library is behaving exactly as it should. 
When re-running another experiment after that on the cluster, I got running time prevision much larger than it should compared to the previous experiment times. I also got a strange error maybe linked to memory leak. For this reason, all the benchmarking will have to be done on the epfl machine of Andrei for consistency.

Since everything is running smootly with the library, it is now possible for me to extensively test other networks (I first want to try with other SimCLR networks architectures, and with the v2 version that i know exists), and all of this on bigger datasets for which the image manipulations will be done on-the-fly (which will incure a large overhead to the whole process but will save a very big disk space). I also plan to investigate other distances to see the difference with cosine and Jensen-Shannon.


## Week 7 : 21/03 <a name="week7"></a>

This week, I performed several experiments on other models, especially other SimCLR models (v1 and v2). The v2 versions provide better results than all v1 models, but the difference between v2 models is not very significative, suggesting that we can use a smaller model and still get almost as good results as with a bigger model. The Jensen-Shannon distance is still better than cosine (and L1 and L2) distance.

I also added support for L1 and L2 (basically any norm) distances. At first, I clipped the distances to the interval [0,1] (by first applying a softmax or normalization by the sum, so that the feature vector sums to 1, and then dividing by $(\sum x_i)^{(1/ord)}$) to be consistent with the other distances (hamming ratio, J-S and cosine), but this yielded very poor results, since the thresholds to actually see an "in between" between a fpr of 0 and a fpr of 1 were so small that it was completely unusable. Thus I decided to remove the normalization, so that the norm are in the interval [0, $(\sum x_i)^{(1/ord)}$]. However, this did not solve the previous problem, and the thresholds were still way to low. Finally, I decided to remove any normalization, and just use the distances as it is, using a statistical study to get the necessary thresholds. This worked (at least on the BSDS500 dataset) with thresholds between [4, 12] for L2 distance, and [100, 250] for L1. However, as expected these distances perform worse than cosine and J-S. 

I then tried to improve the J-S computation across batches to remove the overhead, but did not succeed. I first tested different implementations that would be used in the same spirit as it is used now (just apply the distance one after the other), but it seems that speed cannot be traded for robustness in this case (otherwise a 0 or very low value in the normalized feature vector will yield big inacurracies). After that I explored way to process everything in "one pass", but it is not as trivial since we need to compute every pairwise distances between the database and the current batch. One function from scipy does this and provide a speed-up of about 20-30%, but this is not as good as hoped and does not justify a change in the current framework in my opinion. Trying to use pytorch to compute everything in one pass is not obvious. This could be investigated further but would need to be quite smart about the implementation. 

On the results part, the same networks as SimCLR uses (basically big ResNets) but trained in a supervised fashion on ImageNet perform worse than the constrative approach used in SimCLR, as expected. Moreover, the neural models from SimCLR do not seem to have the same "weaknesses" (in the attack-wise performances sense) as classical algorithms like Dhash or Phash, suggesting that a smart ensemble of these methods could (I believe) provide quite an improvement to the robustness of the final algorithm. This is further motivated by the fact that classical algorithms (Phash or Dhash for exemple) have a fpr of 0 up to quite high recall, meaning that we could increase the overall performances without incurring any drawback to the final algorithm.