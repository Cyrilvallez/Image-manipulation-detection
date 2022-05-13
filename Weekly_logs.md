# Weekly logs

These are logs that I will update every week in order to keep track of my work (and possibly so that other are also able to follow it).

## Table of contents
1. [Week 1 : 07/02](#week1)
2. [Week 2 : 14/02](#week2)
3. [Week 3 : 21/02](#week3)
4. [Week 4 : 28/02](#week4)
5. [Week 5 : 07/03](#week5)
6. [Week 6 : 14/03](#week6)
7. [Week 7 : 21/03](#week7)
8. [Week 8 : 28/03](#week8)
9. [Week 9 : 04/04](#week9)
10. [Week 10 : 11/04](#week10)
11. [Week 11 : 18/04](#week11)
12. [Week 12 : 25/04](#week12)
12. [Week 13 : 02/05](#week13)
12. [Week 14 : 09/05](#week14)

   


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

## Week 8 : 28/03 <a name="week8"></a>

This week, I first tried to get the same results as before on a bigger dataset, namely ImageNet validation set, containing 50 000 images. However, I first got some bugs on the generation of text attacks. Indeed, some images on ImageNet have strange sizes, on which the currently calculated text size would result on a textbox too big. For this reason, I first add to modify the generation of text attacks. This took me quite some time to manage to obtain a coherent text size for different image sizes (in the sense that the text should occupy the same amount of space in two images of different sizes), especially since font size are not exactly linear. Going from text size 12 to 13 is not the same as going from 23 to 24 for example. Finally I managed to get robust and coherent text generation. 

After this was solved, the problem of efficient distance computation became the biggest problem : indeed in the actual state the time needed for the experiment was way too large. Indeed, we are looking at 116 000 thousand variations of images, and we need to compute the distance to 50 000 images for each of the 116 000. And this without taking into account the time needed to get the hash, nor the fact that we need to perform this for several algorithms. For this reason, I had to find a way to compute the distances on GPU as investigated before. In the end, I used pytorch for this while changing some data structures, which allowed to compute the distances between batches and the whole database in one go on GPU. After this was done, the time needed for the experiment was still large, but acceptable (about 10 to 100 faster than before). 

Finally I got some results with ImageNet dataset. Everything seems to be similar as what I got before on the BSDS500 dataset, except with slightly worse performances, which was expected as we process and compare to more images.

## Week 9 : 04/04 <a name="week9"></a>

This week was mostly used to advance on the benchmarking paper. I wrote all the method and started working on the results. After some reflexion I also adjusted some parameters on the attacks, in order to make more sense overall. I thought about the most appropriate representations of the results and performed different experiments to compare different ways.

## Week 10 : 11/04 <a name="week10"></a>

Once again, this week was devoted to the paper. I performed different comparisons of hash length, number of features, distance metric... It took quite some time to choose between different representations of things, adjust size for the figures, decide what to show or not,... I performed a study on the performances depending on the database size, showing that performances degrade when the database increases (more false positive). I also studied the behavior on the memes dataset to check performances on "real" data.

## Week 11 : 18/04 <a name="week11"></a>

Week of holydays !

## Week 12 : 25/04 <a name="week12"></a>

This week was mostly used for advancing the paper as well as reading litterature related to image retrieval using deep learning. Particularly, I inspected everything related to the memes dataset since we had some issue and I did not have all the images (about 2000 out of 46 000). Some manual data exploration highlited that the dataset is quite dirty. Some memes are absolutely not the same as the template they are supposed to be a variation of. This is notably the case for the "zuckerberg" memes which are all very different from one another and to the corresponding template. Moreover, some templates correspond to a version of a meme that is already quite changed from the original, which is a problem to find the other corresponding memes. Finally, some memes are an "aggregation of different templates" in the sense that they are memes used in other memes. But they are supposed to be mapped to only 1 template, which is an issue if a meme in the control group (thus its template is not in the database) is an aggregation of its own template and a template in the experimental group. Indeed, it is likely to be detected in this case, but will be counted as false positive while the matching was correct in a sense. 

Moreover, it appears that thresholds in the distance metric obtained with the BSDS500 dataset do not extend at all to the memes dataset in order to get similar false positive rate. For example, the threshold set on the BSDS500 dataset to get 0.005 fpr for SIFT gives about 0.5 fpr on the Kaggle dataset. Thus comparing by only number of detection using the thresholds obtained on the previous dataset will be all except a fair comparison since SIFT (the algorithm for which it was the most critical) will give the false impression to perform quite well when half the detection are in fact wrong. My guess is that it has something to do with the text which is present on all images in the Kaggle dataset. Indeed, the detector may be finding edges of the text zones or the letters, which are in fact present on every images.
Deciding on the thresholds is indeed something that we will have to do prior to deploying the system, because we won't have access to groundtruth at this time, but it should be done on a more representative dataset of the images we will get in the wild to avoid the same kind of problems.

Apart from that, I read about DINO which is a self-supervised training technique for vision transformers that seemed very promising. I implemented it in the current framework, but tests on the BSDS500 dataset showed that it was performing very well, but not better than SimCLR v2. I will not push the investigation at this point since the goal is to move to the database search at scale, and not to pursue experiments on detections. I read papers on image retrieval, notably 

- Neural Codes for Image Retrieval
- Nested Invariance Pooling and RBM Hashing for Image Instance Retrieval
- CNN Features off-the-shelf: an Astounding Baseline for Recognition
- Visual Instance Retrieval with Deep Convolutional Networks
- Large-Scale Image Retrieval with Compressed Fisher Vectors

and some other that I just quickly looked at. 

## Week 13 : 02/05 <a name="week13"></a>

This week, monday and tuesday were used to make my slides and prepare for the student exchange presentation. Then on wednesday, I cleaned most of the main Github repository, wrote the Readme etc... for when we will submit the paper. Finally, the end of the week was devoted to read and discover how ElasticSearch and Faiss work for similarity search. After some time, it appeared that faiss would be easier to use and integrate into Python code, and is thus the preferred direction for now. There is no clear documentation, but some examples on their Github. 

## Week 14 : 09/05 <a name="week14"></a>

This week, the first trials with Faiss were performed. I first downloaded half of the [Flick1M dataset](https://press.liacs.nl/mirflickr/), resulting in 500K distractor images to emulate large scale image search, following what we discussed during the meeting the week before (we agreed that millions of images were not needed, and that we would go to the hundred thousands). Then I created a new Github repository that start this new part of the project, on which I added code to extract the features on the different datasets and save them to file for fast future access. I first had a lot of memory issues when dealing with such large arrays. At first, the problem seemed to come from Pytorch dataloader and the workers for subprocesses, but finally appeared to be due to numpy implementation of their function and copy process when creating new arrays. After this trouble, I was able to really start testing Faiss on the data. 

Of course, I first tested with brute-force matching, to get a baseline both in term of search time and accuracy we could hope to obtain. As accuracy metric, I decided to use recall@k measure (proportion of target images present in the k nearest neighbors of the query images) which in my opinion make sense and seems to be used in this kind of context. I first tested with recall@1, meaning we only look for 1 nearest neighbor of each query. 

The first observation is that Faiss is incredibly fast. When looking at 1 nearest neighbor, it takes only about 10-15s on GPU to get the results for ~40K queries on a database of about ~500K. This is at least the case for L2 and cosine distances (L1 is a bit longer, and Jensen-Shannon way more --> about 800s). The second observation is that we get pretty good results from the neural descriptors (SimCLRv2 Resnet50 2x) : the recall@1 for brute force cosine similarity is 0.806 on the Kaggle memes dataset (but remember : this dataset is pretty dirty !) and 0.95 on the artificial attacks BSDS500 dataset ! Of course, since this is brute-force we won't be able to improve those recall number, only the time needed to get results (may not even be true -> sometimes PCA improves on the baseline brute force results). However, checking for more neighbors (recall@5, recall@10,...), should improve results. In the end it seems that last months efforts were not for nothing. 

I also compared the brute-force baseline to clustering and searching only to the nearest clusters. Of course this improves the time needed for the search, but does affect performances. However, when searching a relatively high number of clusters (50 to 100), we can get faster results without loosing much performances, which is a good start. I now want to experiment in priority with PCA for dimensionality reduction, since this may drastically improve search time while not affecting too much the recall. Then an analysis of the clusters would be nice to get a sense of how the data is partitioned. So those are the next objectives, along with refining the current framework for easier benchmarking of techniques.