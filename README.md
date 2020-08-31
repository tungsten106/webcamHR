# webcamHR

```python
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
```



### Heart rate detection

The main propose of [webcam-pulse-detector](https://github.com/thearn/webcam-pulse-detector) is to use camera for heart rate detection. Its code provides the function that measures the color variation of a rectangle on your forehead to generate a HR value on screen. I first starting to adjust the original code is because it always forces close before the experiment, which is solved by throwing expectation.

However, while continuing the experiment, we found that forcing close is not the main problem. Although the result generated by the author was successful, the BPM output of our experiment was unstable and incorrect. (Later I found that the code he used was a older version with use of openmdao, while ours is the simpler version without that package.) To improve the accuracy on my device, I read several articles before changing the code.

Before the article review, I will first introduce the theme of camera pulse detection and some analysis of original code.



### Heart rate detection using camera

In traditional medical environments, it always requires contact to patients' skin to measure the heart rate, such as electrocardiogram (ECG), adhesive gel or short-term measurements through fingers on wrist. But those methods might be not be comfortable or causing friction on skin, and may have hygiene concerns especially during the pandemic. Thus, contactless measurements are helpful. From 2005 there were researches about that, which detecting variation of skin color through deliciated light(e.g. infra red). Since the absorption of blood and surrounding skin is different, the changes of color when the heart beats are visible under cameras. In 2008, Wim Verkruysse et al.([Remote plethysmographic imaging using ambient light](http://www.opticsinfobase.org/oe/abstract.cfm?uri=oe-16-26-21434), which is also referenced by author of webcam-pulse-detector) proposed that it is possible to use ambient light instead, and camera to record the pixel value on the screen. They found that the best region of interest(ROI) is a rectangle on forehead, and the green channel have the best result for HR estimation. This result agreed by most following researches, including the articles I viewed.

Basically, the experiments follow steps as follows: Face detection - selecting ROI - get the average pixel value(PV) - data processing on PV for better evaluation - using FFT(Fast Fourier Transform) to get the frequency of variation of PV - HR estimation as output.



### webcam-pulse-detector

When running with the command shown in original Github, a camera window presents and there are several functions. 'C' is for changing cameras, in case that there are more than one camera on the computer, or the user is using external connected cameras; 'S' is for searching faces and start detection. After selecting S, there are D: showing graph of BPM and F: printing acquired data in csv from video. (In the original code, it prints pixel value(PV)/time, and I changed it to BPM/time.)

When the experiment starts(S), the camera locates face position with OpenCV. A rectangle on forehead will be fixed and keep recording the average pixel value change, as demonstrated in article he referenced.(Wim Verkruysse et al.) But although in the article it's said that green channel have best illustration to color change, there wasn't much difference when I adjusting the channel involved in HR estimation.

The PV(t) collected from camera will be interpolated first, with a mean time interval from beginning, and then standardization(minus mean). For a more accurate result of frequency extraction, the data will times hamming function(to make the waves more obvious). Then [FFT in numpy](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html) is used to extract frequency of signals, which output is in form of complex numbers. Take module of those complex numbers and find the index of highest amplitude, which represents the coefficient of PV in sample frequencies. Sample frequencies are generated from numbers of acquired data/passing time, which is a list of time that recording each data from the beginning of the experiment. Since the function he used is fft.rfft(), which returns the part without conjugates. Thus, the length of output is halved. To correspond with the output, the length of sample frequencies also have to be the same size. As a result, sample frequencies is halved to be the time at recording each two data from beginning.

Since the experiment is about heart rate, the frequency value might within a range. The author select the frequency that makes BPM within (50,160) to restrict the data (just times 60). And the result of FFT is also sliced with corresponding index. Finally the frequency with coefficient of highest modules will be selected, which is number of heart rate per second. it would be BPM if times 60.

As a result, a BPM is generated and it will be printed on screen.



### Article review

#### [Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. (Ming-Zher Poh et al.)](osapublishing.org/oe/abstract.cfm?uri=oe-18-10-10762)

Poh et al. suggest that, tranditional experiments of pulse detection through camera are "lacked rigorous physiological and mathematical models amenable to computation". This caused the problem that the measurements might still be affected by noise. In order to present better result, Poh et al. used ICA(Independent component analysis) to extract original signal from observed ones. It supposed that observed signal $\textbf{x}(t)$  has a linear relationship with the original signal $\textbf{s}(t)$. Both $\textbf{x}(t)$ and $\textbf{s}(t)$ are 3-d vectors with each component represents pixel values of each RGB channel. They are related with a 3*3 matrix:

$`\sqrt{2}`$

$x_i(t) = \sum_{j=1}^{3}a_{ij}s_j(t)$ for each i=1,2,3

Let's represent matrix with elements $a_{ij}$ as $\textbf{A}$. The purpose of ICA is to find the matrix $\textbf{A}$ to estimate the original signal. That is: $\textbf{x}(t)= \textbf{A} \textbf{s}(t)$. In order to get the estimation $\hat{ \textbf{s}}(t)$, the inverse matrix $\textbf{W}$ need to be calculated. Therefore,

$\hat{ \textbf{s}}(t)= \textbf{W} \textbf{x}(t)$

In the experiment of Poh et al., they used[joint approximate diagonalization of eigenmatrices (JADE) algorithm](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.308.8611&rep=rep1&type=pdf) to find the inverse matrix $\textbf{W}$.

The process of experiment can be explained by a figure from article. It used camera on computer (Macbook Pro) to take the video of participants, that is approximately 0.5m from the computer. In order to evaluate the accuracy of estimation, the "ground truth" of heart rate at same time of video is collected from a sensor on fingertip.

<img src="C:\Users\yexue\webcamHR\Untitled picture.png" style="width:400px;height:250px;">

When the camera recognized face with OpenCV, whole face is selected as ROI. Then pixel values(PV) in ROI in 3 channels are collected and normalized ($x'_i(t)=\frac{x_i(t)-μ_i}{σ_i}$, where $μ_i$ is mean and $σ_i$ is standard deviation). $x'_1(t),x'_2(t),x'_3(t)$ from each channels generate estimation of original signals $\hat{s_1(t)},\hat{s_2(t)},\hat{s_3(t)}$ with JADE algorithm. Since green channel shows the clearest frequency, and for the simplicity, they decided to always use component 2 for heart rate estimation.

Finally, Poh et al. used FFT for frequency extraction as well as Verkruysse et al.. Other from the application of ICA, they also compared the historical value with new estimation to reduce the influence of noise. If the difference between last and new value exceed the threshold (12 bpm), the next highest peak will be selected. If there isn't value that satisfied, they will retain the last value.

From the result, the application of ICA did improve the accuracy of estimation and decreased the deviation of data from mean. The difference with respect to correct value from fingertip sensor, 95% of differences are between -4.55 to 4.44 bpm. The accuracy compared with the difference without ICA, which is about 10 bpm, is obviously higher.



#### [A Machine Learning Method to Improve Non-Contact Heart Rate Monitoring Using an RGB Camera (Ghanadian et al.)](https://ieeexplore.ieee.org/abstract/document/8478131/)


Ghanadian et al. suggests that there are few limitations in previous experiments: 1. there are restriction on movements, 2.too sensitive to light noise, 3.methods to select the best component from ICA output are "unproven heuristic methods". For example, in the other article, Monkaresi et al. claims that the third component reflects the heart rate the best. Thus, this article proposed a method to select ICA output component with help with machine learning. For first problem, Ghanadian et al. use light equalization scheme reduce the influence of light noise on pixle values detected. Moreover, the author used linear regression model to improve accuracy at last step of HR estimation.

<img src="C:\Users\yexue\webcamHR\Untitled picture-ML.png" style="width:380px;height:400px;">

Ghanadian et al. inserted and changed a few steps based on previous experiment processes. OpenFace is used to find ROI (which is full face like that in Poh et al.). Then light equalization is used. To be more specific, pixels in ROI are decomposed into Lightness(L), Saturation(S) and Hue(H), and L is normalized while others unchanged. Normalized L' will reduce light fluctuation and the face might seems darker. After detrending and normalizing, average PV from video in RGB channel are analyzed by ICA. As demonstrate by the flow diagram, signals are filtered to reduce noise. Low pass filter is used for weakening signals outside the cut-off frequency; Hamming window is used to make the waves more obvious so that the frequency is more accurate; Butterworth filter is used to "clean up signal".

The component selection is using machine learning. 5 features are extracted by PCA. The first 3 are in frequency domain: the amplitude, frequency and ratio($\omega=\frac{max \lvert {y(t)} \rvert^2}{E[\lvert {y(t)} \rvert^2]}$, $y(t)$ is the frequency) of the highest peak. The later 2 are in time domain: the mean and standard deviation of time differences between peaks.

By comparing different classification models, Ghanadian et al. found that KNN, Random Forest and K-star presents the best estimation. The author chose combination of later 2 model to improve the accuracy. Input of the model is 5 features of signals, and the output are evaluation of estimation of each componentsthat whether it gave a correct result(Y/N). The HR estimation will take average if there are more than one YES in output, and so if all output are NO. Finally, to reduce the linear shift between estimation and the correct value, Ghanadian et al. used linear regression as the last step of estimation.

The result show that ratio of component in the final output are 13.11%,30.58%,56.31%. Its accuracy was improved from previous experiment (Poh&Monkaresi), and LR reduced the standard deviation.