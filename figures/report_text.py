# -*- coding: utf-8 -*-
"""
report_text.py
======================================================================
Prose for the project report.

Abstract, Introduction, Methods and Results up to and including 3.2 are
the author's own revised text, captured verbatim from the edited
manuscript. Sections 3.3 onwards live in report_text_rest.py.
"""

from report_text_rest import RESULTS_REST, DISCUSSION, CONCLUSION

TITLE = ("Automatic eyelid exclusion by deep-learning segmentation enables "
         "ocular torsion measurement in close-up eye video")

AUTHOR_BLOCK = [
    'Sohil Ananth',
    '',
    'MSc Bioinformatics and Computer Science',
    'Department of Genetics, Genomics and Cancer Sciences',
    'University of Leicester',
    '',
    'Supervisor: Dr David Souto',
    'Division of Psychology and Vision Sciences',
    '',
    'BS7130 Independent Research Project',
    'August 2026',
]

ABSTRACT = [
    'Ocular torsion, rotation of the eye around the line of sight, is the most difficult degree of freedom to measure among the three rotations. It cannot be measured via pupil segmentation since a rotating iris looks the same way; thus, torsion can only be obtained through analysis of iris texture. The video-based techniques for doing so can be categorized into feature-tracking and cross-correlation using polar unwrapped image of the iris against reference. They both require visibility of the iris; however, eyelids violate this assumption in case of close-up videos.',
    'In this project, the inference-only pipeline was constructed based on RITnet, a deep-learning eye segmentation network, combined with classical feature tracking eye tracker and was tested against the independent open-source eye tracker OpenIris. Segmentation provides three advantages over the classical feature tracker: knowledge of the iris location, information whether the eye is closed or not, and distinguishing of the iris from the eyelash. A single 9.4-minutes-long recording (28,236 frames at 50 Hz) was analyzed',
    'The reliability of measurement was calculated using split-half reliability, where torsion was calculated separately from two randomly chosen halves of the tracked features in each frame and correlated between them. Substituting frame-by-frame calculation of Lucas-Kanade chains with tracking relative to the segment reference image improved reliability from 0.486 to 0.768 and decreased the standard deviation of noise from 0.262 to 0.106 deg. Residual from a fit of a rigid rotation ceased to increase through a segment, changing from 2.8-fold increase to 1.5-fold.',
    'OpenIris with the same settings applied to the same images detected the pupil with an equal efficiency (89 per cent compared with 89 per cent), but generated much less reliable trajectory, with horizontal displacement of the pupil frame to frame of 13.6 px compared with 0.76 px, and autocorrelation of torsion at lag 1 of 0.06 compared with 0.80. Activation of its eyelid tracking using Hough-lines made no difference. This was due to the measurement of the mask. 59.5 per cent of the area cross-correlated were unoccluded iris, decreasing to 24.2 per cent in the upper part of the annulus.',
    'However, this measure is constrained by lid occlusion rather than camera resolution, while segmentation-based gating eliminates this constraint. Control analyses revealed absence of direction-specific torsion as well as lack of directional torsional drive, which should be seen in recordings not exposed to any kind of rotational stimulation. The value of reliability stays at the internal maximum because there was no valid trace of torsion from OpenIris analysis for this recording.',
]

INTRO = [
    ('p', 'Vision relies upon movement. The retina adjusts to a constant stimulus within a second or so, and hence a stabilized image will be lost from consciousness, and the eye is never completely still, even during the process referred to as fixation. Between each major saccade, which reorientates the line of sight, the eye makes smaller movements of three types: micro-saccades of a few tenths of a degree, slow drift between each micro-saccade, and a low amplitude tremor. Such eye movements are specified in terms of arcminutes, thereby posing problems for instrumentation that would otherwise be unnecessary in the case of a conventional eye tracker.'),
    ('p', 'The problem of measuring small eye movements is thus one of establishing what the instrument can detect, and this poses difficulties since both the quantity, and the noise have similar ranges. A standard deviation in degrees of a few tenths of a degree during fixation might refer to the eye or the tracker, and there is nothing inherent in the data to tell which is which. The current research addresses this issue specifically in the context of ocular torsion.'),
    ('p', 'The eye has three axes of movement, all of which rotate the eye relative to some coordinate system. Two of them are horizontal and vertical axes, and they determine the position of the line of sight and are typically measured by eye trackers. The third axis is the rotation around the line of sight itself, and it is referred to as ocular torsion. Ocular torsion contains unique information and is related to the function of the otolith organs, the ocular countertroll reflex, and the restraints imposed on eye orientation by the oculomotor system; it varies in ways in vestibular and strabismic pathology.'),
    ('p', 'Ocular torsion is, however, the most difficult of the three to measure. The explanation for this is geometric rather than technological. Rotations in the horizontal and vertical axes change the position of the pupil within the image, and hence the method that detects the position of the pupil measures the two. A rotation in the torsional axis does not change anything because the pupil is round and coaxial to the iris, and thus the eye can rotate around the line of sight with each pixel boundary staying in place. All that remains is the detection of the rotation of the iris texture.'),
    ('p', 'There are two families of video methods for this. One is the sparse approach which involves tracking image features in the iris from one image to the next, often through corner detection and optical flow (Shi and Tomasi, 1994; Lucas and Kanade, 1981). Then, it calculates a rotation from the displacements of the tracked features. The second family transforms the iris annulus to polar coordinates and then averages along the radius direction to get one-dimensional intensity signature that it cross-correlates against a reference frame to compute the angular shift. Correlation methods are descendants of the deformable iris models of the 1990s (Ivins, Porrill and Frisby, 1998) and have become popular in modern iris recognition algorithms such as OpenIris (Sadeghi et al., 2024).'),
    ('p', 'These two families have different failure modes, and this makes a difference for the following. A feature tracker is a sparse method that selects a couple of hundreds of points, tracks each point and then pools displacements and hence has a freedom to refuse placing a feature somewhere it feels like and give an estimate anyway. On the other hand, the cross-correlation is a dense method that computes one signature over the whole annulus and then matches it as one entity and thus it doesn’t have a built-in way of rejecting some sectors and thus everything goes into the correlation peak regardless of whether there is iris in the sector or not.'),
    ('p', 'Both families of algorithms operate under an implicit assumption that is hardly ever stated: there should be an iris visible in the region of the image.'),
    ('h2', '1.1 The eyelid problem'),
    ('p', 'In eye images, the height of the palpebral fissure is less than the width of the iris. It is not a problem specific to any system used for the purpose; it is a biological fact. The upper eyelid, along with its lashes, covers the upper portion of the iris, while the lower eyelid covers only a part of the lower section of the iris; hence, a circular region, which is large enough to contain the limbus region, also contains non-iris tissue. The extent of the problem changes with the extent to which the eye is open, and the variation exists in every recording as well as between individuals.'),
    ('p', 'The problems are different for each family, although both are serious. Eyelashes are high-contrast objects, and a corner detector is likely to detect them instead of the relatively low-contrast structure of iris stroma. Features are seeded unconstrainedly in great numbers on lashes. However, eyelashes move in conjunction with the eyelid and not with the eyeball; hence, lid movement becomes part of the rotational movement of the eye. About the correlation methods, the situation is quite different. Polar signature is determined all along the annulus, and any segment covered by the eyelid will contribute a pattern, which rotates independently of the eye movement.'),
    ('p', 'The classical version of irissometry employed for this work does not shy away from labeling the challenge. According to its source code, automated eyelid detection is still one of the open issues, with the comment to remove the features of the eyelids. OpenIris provides an option to track the eyelids, though it is not turned on by default.'),
    ('h2', '1.2 Segmentation as a source of anatomical labels'),
    ('p', 'Convolutional networks that are trained to segment eyes can provide what classical approaches lack – an anatomical label per pixel. RITnet (Chaudhary et al., 2019) is an encoder-decoder network, 248,900 parameters, which labels pixels of an eye image as belonging to the background, sclera, iris, or pupil. It won the OpenEDS Semantic Segmentation Challenge and is faster than the frame rate of almost all eye trackers.'),
    ('p', 'Four-class label may seem of limited use for calculating torsion, but the truth is that iris label is not a pixel that was detected as being the eyelid, a lash or a sclera pixel. This is exactly the kind of information that a feature tracker needs to avoid tracking of lashed areas, and this comes as a bonus when segmenting is performed anyway. The pupil label gives another piece of information that classical approach lacks – the blink detection: when there is no pupil segmented, the eye is closed, and no need to count features.'),
    ('p', 'Deep learning is used here strictly for inference. No network was trained or fine-tuned. The published RITnet weights were used as released, and the contribution of this project lies in how segmentation output is coupled to a torsion measurement, not in the network itself.'),
    ('p', 'This choice has a price tag attached to it, and it became apparent almost instantly. A trained model on one distribution of images cannot always be used on a different distribution, and the RITnet model was trained on OpenEDS where the eye takes up only a small portion of the frame, and there are some periocular skin and eyelashes around it. The recordings considered here are close-up shots where the eye takes up the whole frame. Using the model in its raw form led to unusable results. The solution chosen, rescaling and padding the frames to match the distribution in terms of the ratio of the eye within the frame as in Section 2.3, needs to be noted as a lesson applicable beyond this case: the adaptation of a trained segmentation network to new video requires framing statistics even more than good image quality.'),
    ('h2', '1.3 The validation problem'),
    ('p', 'Torsion measurement has an awkward property: there is usually no ground truth. A subject cannot be instructed to roll their eye by a known number of degrees, and scleral search coils, the historical reference standard, are invasive and no longer routinely available. Reported precision figures for video torsion are therefore mostly internal measures, such as the standard deviation during steady fixation, and these are easy to improve for the wrong reason.'),
    ('p', 'The failure mode matters enough to state plainly. Under Lucas-Kanade tracking, a weakly-defined feature does not jump about; it slides smoothly along an intensity gradient. Its error accumulates gradually and its frame-to-frame difference stays small. Any metric based on smoothness will therefore reward the change that produced it, while the estimate drifts further from the truth. Preliminary work on this project reproduced the effect: lowering the corner quality threshold reduced frame-to-frame jitter from 0.167 to 0.148 deg and increased drift within a segment from roughly 1 deg to 15 deg.'),
    ('p', 'The same reasoning applies to any measure computed from the output series alone. Within-segment standard deviation falls if the estimate is smoothed. So does frame-to-frame jitter, by construction. Drift within a segment is more resistant, because slide accumulates and shows up as a difference between the start and end of a segment, but it is still a property of one series and cannot separate a real rotation from a systematic error that happens to be smooth.'),
    ('p', 'Two checks were used here to avoid that trap. The first is internal but cannot be improved by smoothing: split-half reliability, computed by splitting the tracked features into two random halves, estimating torsion independently from each half of the same frame, and correlating the two series. A real rotation appears in both halves. Noise particular to individual features does not. The second is external: run an independent implementation, with a different algorithm, over the same frames and ask whether the two agree.'),
    ('h2', '1.4 Setting of the present work'),
    ('p', 'This project began from two existing pieces of software rather than from a blank sheet. The first is RITnet, distributed with its trained weights under a permissive licence. The second is an implementation of irisometry in the Strauch and Naber lineage, obtained through collaborators at the University of Applied Sciences Upper Austria and Utrecht University, which performs feature-based torsion tracking and requires the limbus to be outlined by hand before each run. Neither was built with the other in mind.'),
    ('p', 'The observation that motivated the work is that each supplies what the other lacks. Segmentation recovers the position and extent of the iris on every frame but cannot see torsion, because a rotating iris occupies the same pixels. Feature tracking measures torsion but must be told where the iris is, must guess when the eye has closed, and has no way to avoid placing features on the eyelashes. Coupling the two is therefore not merely convenient. It removes three distinct limitations of the classical method using information the network already produces.'),
    ('h2', '1.5 Aims and hypotheses'),
    ('p', 'The project had two aims. The first was to build an inference-only pipeline that couples RITnet segmentation to classical feature-tracking irisometry, such that segmentation supplies the region of interest, the blink signal and a per-frame iris mask. The second was to establish whether the resulting torsion estimate is a measurement rather than an artefact, by internal reliability and by comparison against OpenIris.'),
    ('p', 'Three hypotheses were tested.'),
    ("bullet", [
        'H1. Restricting feature detection to segmented iris tissue improves the reliability of the torsion estimate relative to a circular region of interest alone.',
        'H2. Tracking features from the segment reference image, rather than chaining frame to frame, prevents the accumulation of correspondence error and improves reliability.',
        'H3. On close-up footage with substantial lid occlusion, a method without lid exclusion produces a less stable estimate than one with segmentation-based gating.',
    ]),
    ('p', "A fourth question was posed at the outset and answered negatively during the work. Whether measured torsion varies with gaze direction in the manner Listing's Law predicts cannot be tested from this recording, because gaze direction barely varies within it. That analysis is retained as a control rather than as a test, and the reasoning is given in Section 3.8."),
]

METHODS = [
    ('h2', '2.1 Recording'),
    ('p', 'Several videos were examined (6-8 videos in total) and one video sequence was chosen; this will be called Video 8 for the rest of this report. Video 8 is 908 x 620 pixels at 50.0 frames per second and 564.7 seconds long, resulting in a total of 28,236 frames. Video 8 is recorded using infrared lighting and is close-up, such that the eye occupies much of the frame. Horizontal visible iris size is 382.9 px (median) in the segmentation results; since we assume a corneal diameter of 11.71 mm, we can say that there are 32.7 px per mm. The corneal diameter we assume (11.71 mm) is the mean value in healthy adults according to Rufer, Schroder and Erb (2005); 11.71 +/- 0.42 mm is the standard estimate of the horizontal visible iris.'),
    ('p', 'Seven additional videos have been acquired and were not processed in this analysis. The problems associated with working with only one video sequence are discussed in Section 4.5'),
    ('h2', '2.2 Software and availability'),
    ('p', 'All code for data analysis in this project is available at https://github.com/SxR24/Micro-analysis-of-eye-movements-using-deep-learning-neural-networks. In this git repository one can find the whole pipeline, scripts for figure generation which were used to create all figures in this document, and the scripts for data analysis that generate all statistics mentioned in this paper. Licences for third-party software are listed in THIRD_PARTY_NOTICES.md file. RITnet was used according to the MIT Licence (Copyright 2019 Chaudhary et al.). Its model and weights were not modified.'),
    ('p', 'The pipeline is developed in Python 3 using NumPy, OpenCV, pandas, Matplotlib and PyTorch. Comparison with OpenIris was performed using 0.1.5 (Windows) version. Video recordings were not provided with the code.'),
    ('h2', '2.3 Frame extraction'),
    ('p', 'RITnet was trained using OpenEDS, where the eye takes up a relatively small portion of a 640 x 400 image, with periocular environment surrounding it. Used straight away on the footage, where the frame is completely filled by the subject, RITnet failed to produce any usable masks. In order to make sure that each frame was reduced to a size similar to the training data, each frame was resized and centre padded to fit a 640 x 400 image, rather than stretching the image, since this might distort the iris.'),
    ('p', 'For this purpose, the scaling was done such that 85 per cent of the image area is covered by the contents of the frame. For video 8, the source resolution 908 x 620 is mapped to 498 x 340, with a scale of 0.5484 and horizontal and vertical padding of 71 px and 30 px respectively. The values are saved in the metadata file for use in subsequent stages:'),
    ("code", '    original = (mask_coordinate - pad) / scale'),
    ('p', 'The segmentation output is thus in padded 640 x 400 mask space, whereas the torsion is measured in original video space, and there needs to be an explicit matching wherever the two come into contact. This is the one easy mistake to make in the whole process, whereby a small offset in coordinates makes the output seem entirely plausible.'),
    ('h2', '2.4 Segmentation and geometry'),
    ('p', 'Inference was conducted on the generated frames using the best_model.pkl weights described in the RITnet literature, generating four-class masks for each frame. The measurements of the pupil and iris geometries were then derived from these masks.'),
    ('p', 'Two aspects of the segmentation result make the simple measurement procedure inaccurate, but both have been accounted for. First, the iris class does not correspond to a disc, but rather an annulus, as the pupil class is separate; a diameter calculated using only the iris class underestimates the true diameter considerably, so all measurements are done on the combined iris and pupil classes. Second, the eyelid margins cut off the iris from above and below, and very rarely to the same degree, making the vertical extent of the visible portion of the iris not equal to the actual diameter, and shifting its vertical center towards whichever lid occludes less. In this video, the upper lid obscures roughly 38 pixels while the lower only 14 pixels.'),
    ('p', 'The diameter of the iris is thus defined by its horizontal dimension alone as the median of the 10% widest occupied rows, in order to avoid any contamination of the measurement due to spurious segmentation pixels that may occur within a bounding box. The horizontal center point is based on the same set of rows while the vertical center point is calculated using the pupil center whenever the pupil is available.'),
    ('h2', '2.5 Region of interest'),
    ('p', 'In the original implementation of irisometry, the limbus must first be manually marked during the initial stage of each test. Here the region of interest will be obtained based on the segmentation process. The position and size of the iris will be translated into the original image frame, and the median will be calculated among all those images in which the iris is identified, excluding images where the iris diameter measured is greater than three median absolute deviations from the median.'),
    ('p', 'The region of interest will remain constant throughout the test. A different region of interest on a per-image basis will mean that the object that we are comparing the torsion to will change, and torsion is a rotation against an immobile point of reference.'),
    ('h2', '2.6 Torsion tracking'),
    ('p', 'The Shi-Tomasi corners are detected within the region of interest and are tracked using pyramidal Lucas-Kanade optical flow technique. There are three changes in the algorithm, all resulting from the failure of the algorithm to perform its tasks effectively.'),
    ('h3', 'Feature gating by the segmented iris'),
    ('p', 'The detection is hidden using the RITnet class of iris for that frame, which is mapped back to the original video coordinate space and subjected to an erosion of 6 px. The erosion strips off the edge of the pupil, the limbus, both of which constitute strong corners and are not part of the iris texture, as well as the edge of the pupil, which changes due to dilation but not due to rotation. Also excluded are near saturated pixels greater than 248 via a dilation of 3 px, as reflections from the infrared illuminator are constant and do not change due to rotation.'),
    ('h3', 'Anchoring on the reference image'),
    ('p', 'The straightforward approach passes optical flow calculations through the chain of frame pairs. Every step in the chain introduces a small correspondence error; the errors accumulate over time and can be measured by the increase of the residual from fitting a rigid rotation within the segment (Section 3.2). In contrast, optical flow is calculated between the reference image of the segment and the current frame, while the chained position serves only as an initial guess. The intensity anchor point remains fixed in the reference; thus, the problem of accumulating error in slide cannot occur. Since the optical flow is determined for the whole reference-to-current shift, four pyramid layers and the forward-backward test tolerance of 2.0 px are used.'),
    ('p', "The feature validity is preserved throughout frames rather than recalculated each time. A correspondence that fails forward-backward test for three consecutive frames is removed permanently; features that fail in any single frame are not included in that frame's processing at all. The attempt to retire the features after the first failure proved impractical: one blurring frame removes many good features and reduces the average segment length to 4.5 frames."),
    ('h3', 'Estimating the rotation'),
    ('p', 'The torsion is computed as the least-squares rigid rotation that maps the reference positions of the feature onto the current one. They are both expressed in the centroid-referred coordinates of the same surviving subset prior to the fitting, thus eliminating the translation introduced by the change in gaze and ensuring that the origin shift due to dropping some features is equal for both frames and therefore cancels out. It is the orthogonal Procrustes rotation that is found using closed-form computation and then weighted using Tukey biweight function in five iterations.'),
    ('p', 'It replaces the use of circular median of individual angles of change between features employed previously. The circular median treats all feature equally, but a feature located at distance r from the center constrains the angle of rotation with accuracy of sigma/r, thus making the features located near limbus more informative than those located near the pupil. The Procrustes solution is the maximum-likelihood rotation assuming isotropic Gaussian noise in position measurements and is weighted with the squared radius implicitly.'),
    ('p', 'The accuracy of the estimator was verified using synthetic data for rotations of known magnitude subjected to arbitrary translation, feature dropout in an asymmetric fashion where no more than 15% of the total number of features were dropped out, and random displacement of features through uniform noise to an extent of 30%, where features were displaced by +/- 40 pixels.'),
    ('h3', 'Segmentation of the record'),
    ('p', 'Torsion is quantified relative to a reference frame and is cleared every time a new reference is selected, which occurs on every blink and on the point where the surviving features fall below a threshold and must be re-seeded. Re-seed is distinct from blink – the eyelid has not yet blinked. Both are distinguished from each other, and a segment ID is recorded to explicitly identify the segment for downstream processing. Torsion cannot be compared between segments since it is absolute, therefore, all statistics reported here are done per segment.'),
    ('p', 'Blinks are derived from segmentation, not from feature count. The frames where there is no pupil segment are identified as blinks, followed by skipping a recovery window of ceil(0.4 x frame rate) frames, 21 frames at 50 Hz, until the lid passes the iris region. The frames that report a pupil diameter lower than 0.7 times the median recorded pupil diameter in this recording are also marked as bad frames. This identifies the instances where the algorithm detects a tiny sliver of the lash line and reports it as the pupil; 372 such frames are identified in this recording.'),
    ('h2', '2.7 Fusion'),
    ('p', 'The segmentation geometry and the torsion measures are brought together in a frame index table, where RITnet coordinates are converted to original video coordinates, meaning all columns share a coordinate system. This combined table includes pupil geometry and size, iris geometry and size, blink quality flags, three torsion values, rigid fitting error, feature count, and segment ID.'),
    ('h2', '2.8 Split-half reliability'),
    ('p', 'Raw feature trajectories are kept by the tracker; thus, it is possible to reconstruct the estimate without tracking again. Randomly, features are allocated to either of the two halves. Then, the torsion is independently estimated from each half of the frames using the same Procrustes algorithm, both sets are center aligned on a segment basis, and the correlation between the two is calculated. The correlation of the half sets, r, is used as the reliability of the entire estimate through the Spearman-Brown formula, 2r/(1+r); and the variance is decomposed into the signal and noise variance with the signal standard deviation being the total multiplied by the reliability.'),
    ('p', 'Several implementation decisions influence the value and are stated for the value to be reproducible. The features are assigned to halves once, using a pseudorandom number generator seeded to the same value, and this is done for the entire recording rather than on a per-frame basis, since the former could lead to a dependence between the two halves due to the common centroid. For a frame to be used, at least twenty features must remain after filtering, and at least ten in each of the two halves, since otherwise the Procrustes fit to the small number of points becomes dominated by the noise in the fit. Shorter segments than 25 frames are disregarded, since centering within segments leaves virtually no variance in such a short segment and the correlation is thus meaningless.'),
    ('p', 'The measure is computed without using the tracker, through another module containing another instance of the Procrustes function. This is a deliberate choice because a measuring device sharing the code of the object being measured will not detect a bug in the shared component.'),
    ('p', 'There is one quality about the measure worth mentioning before its values are discussed. It is an upper limit rather than an estimate. The presence of anything common between the two sides increases the measure, such as changes in the iris shape due to pupil dilation, and the movement of the mask border, all of which have nothing to do with eye rotation.'),
    ('h2', '2.9 Comparison with OpenIris'),
    ('p', 'Version 0.1.5 of OpenIris was tested off-line on the same video using the JOM method of calculation of the torsion through polar unwrapping of the iris annulus and subsequent cross-correlation with the template. The algorithms of both techniques do not have any common points, except the video file.'),
    ('p', 'The configuration had to be changed as default parameters were calculated for a different geometry of the camera. The scale was 32.7 px/mm measured from the iris, frequency was 50 Hz, iris radius 191 px and annulus width 70 px, thus sampling the region between 121 and 191 px from the pupil center and outside it. Pupil detection threshold was estimated. 33, which should in theory divide the pupil from the iris according to pixel measurements from segmentation masks, failed to detect the pupil in each frame, while 43 detected the pupil in 89 per cent.'),
    ('p', 'The torsion reference template should be taken during the tracking session and saved into the calibration file. Sessions where this was not done yielded zero torsion and are recognizable by the small size of the calibration file. The eyelid tracking was done using two settings, disabled and the Hough-lines setting, in order not to rely on disabling an important function.'),
    ('p', 'OpenIris records the center of the frame as a fallback when there is no pupil detected, having the pupil located in the center of the image with its width equal to the width of the frame. This data looks like the valid one and was discarded specifically, along with the blinks and the frames not tracked, prior to the comparison.'),
    ('h2', '2.10 Statistical treatment'),
    ('p', 'The data samples at 50 Hz have high autocorrelation and are therefore not independent. Whenever there is a confidence interval provided for a regression slope, it results from bootstrap resampling that uses entire segments and not frames, done 2,000 times. The agreement among the techniques is provided in form of a Pearson correlation between the centered within-segment series, alongside the bias and the 95 per cent limits of agreement in the way described by Bland and Altman (1986). The autocorrelation at lag 1 is given separately for each series, since it is not a conflict between two measurements but a comparison between a smooth and noisy series.'),
]

RESULTS_HIS = [
    ('p', 'The results will be presented in the order of execution of the pipeline. Section 3.1 shows that segmentation and geometry worked as expected. Section 3.2 examines hypothesis H2. Sections 3.3 and 3.4 present two component settings which were varied separately, of which one was without effect. Sections 3.5 to 3.7 examine hypothesis H3 both in comparison to OpenIris and through a direct measure of the occlusion it causes. Section 3.8 contains the results of the control analysis.'),
    ('p', 'H1, the influence of gating with the mask, is not tested against a comparable control here. Gating was implemented before the reliability measure was defined, and the stored data for the pre-gated runs were calculated with another estimator and with another tracking strategy, so a comparison with them would mix up three factors. What can be noted, however, is the original measurement that made the gating necessary: With only the circular region of interest, 33.7 percent of tracked features were situated on the iris and 53.6 percent on the lid and eyelashes, while gated all of them fell on the iris. The evidence supporting H1 in this report is thus indirect and comes from Sections 3.4 and 3.7.'),
    ('h2', '3.1 Segmentation, geometry and tracking yield'),
    ('p', 'The segmentation process generated pupils on 26,831 out of 28,236 frames (95.0 per cent). Following the exclusion of blinks, with the pupil diameter gate applied and the criterion requiring a torsion segment to have the torsion segment active, 25,261 frames (89.5 per cent) had a suitable measurement. The tracking split the video into 237 segments, where 202 started at a re-seed after feature loss and not a blink; 120 segments contained at least 25 frames and were used for the subsequent statistical analysis. The median number of features involved in a calculation was 89.'),
    ('p', 'The median iris diameter is 382.9 px and the median pupil diameter is 202.6 px, both using the original video coordinate system. The video contains 2,617 frames labeled as blinks.'),
    ('p', 'There are two geometrical characteristics of interest that should be noted here since they will be referred to later in the paper. The iris has a diameter of 382.9 px in a frame whose width is 908 px. Thus, the eye occupies 42 per cent of the frame horizontally, which means that the image is a close-up according to the training data criteria; hence, the need for letterbox transform (Section 2.3). Also, the pupil is rather big – 202.6 px or 6.2 mm in diameter, which is normal in a dark room under infrared illumination. This results in an iris band that is 90 px wide between the pupil margin and the limbus. That band is the entire signal of interest for torsion.'),
    ('p', 'Blink detection was borrowed from the segmentation process and not from the feature counting process. The segmentation signal had an earlier feature count that was able to detect 41% of the blinks; it disagreed with the segmentation signal on 834 frames where the eye was closed but was still being tracked. By using the signal from the pupil class, the problem is solved at the expense of making sure that the neural network fails in the correct manner, which it does by having a closed eye.'),
    ('h2', '3.2 Reference anchoring improves reliability (H2)'),
    ('p', 'The two tracking strategies were run over the same frames with all other settings held constant, and split-half reliability was computed for each. Figure 1 shows the result.'),
    ("fig", ('fig1_reliability.png', 1,
             'Split-half reliability of the torsion estimate. Left and centre: torsion computed independently from two random halves of the tracked features in each frame, centred within segment, for frame-to-frame chaining (left, 24,846 frames in 78 segments) and for tracking anchored on the segment reference image (centre, 24,939 frames in 120 segments). Points are thinned six-fold for legibility. The dashed line is equality. Right: median residual to the fitted rigid rotation, binned by position within a segment. Growth across a segment is the signature of accumulating correspondence error; a flat profile indicates that the reference anchor is holding.')),
    ('p', 'Reference image anchoring improved half-set correlation from 0.321 to 0.623, and full estimate Spearman-Brown reliability from 0.486 to 0.768. Standard deviation of total torsion decreased from 0.365 to 0.219 deg. When decomposed, the signal component did not change much (from 0.254 to 0.192 deg), while the noise component decreased by a factor of six, from 0.262 to 0.106 deg. Reference image anchoring does not add noise; this is what it ought to do.'),
    ('p', 'The residual profile from Figure 1 captures how this works. In chaining, median residual per feature to a fitted rigid rotation increased from 3.18 px for the first fifth of a segment to 9.04 px for the last one, an increase by 2.8 times. Given the feature radius of 150 px, this corresponds to roughly 3.5 deg of angular scatter per feature. In reference to anchoring case the same profile increases from 1.11 to 1.67 px, an increase by 1.5 times. Feature set loses its rigid rotating character when a segment ages, and reference image anchoring solves that.'),
    ('p', 'Within-segment statistics followed suit. Frame-to-frame jitter decreased from 0.318 to 0.060 deg, standard deviation from 0.558 to 0.170 deg, and drift from 1.038 to 0.352 deg. The number of features contributing went down from 145 to 89, since correspondences are now retired permanently upon loss. Hypothesis H2 is confirmed.'),
]

RESULTS = RESULTS_HIS + RESULTS_REST

SECTIONS = [
    {"heading": "1. Introduction", "blocks": INTRO},
    {"heading": "2. Methods", "blocks": METHODS},
    {"heading": "3. Results", "blocks": RESULTS},
    {"heading": "4. Discussion", "blocks": DISCUSSION},
    {"heading": "5. Conclusion", "blocks": CONCLUSION},
]

ACKNOWLEDGEMENTS = [
    "I thank Dr David Souto for supervision and for the observation that a "
    "recording without gaze excursion cannot test Listing's Law, which "
    "redirected the analysis reported in Section 3.8.",

    "The RITnet model and its published weights are the work of Chaudhary and "
    "colleagues and are used here under the MIT licence. The irisometry "
    "approach follows the method of collaborators at the University of Applied "
    "Sciences Upper Austria and Utrecht University; the implementation in this "
    "repository was written independently and reproduces the purpose of the "
    "original rather than its code. OpenIris is the work of Sadeghi and "
    "colleagues at the University of California, Berkeley.",

    "Generative AI (Anthropic Claude) was used during this project for code "
    "review, for debugging, and for drafting and editing sections of this "
    "report, in accordance with the module's amber classification. All "
    "analyses, results and figures were generated by the code in the linked "
    "repository and were checked by the author.",
]

REFERENCES = [
    "Bland, J.M. & Altman, D.G. (1986). Statistical methods for assessing "
    "agreement between two methods of clinical measurement. The Lancet 327, "
    "307-310.",

    "Chaudhary, A.K., Kothari, R., Acharya, M., Dangi, S., Nair, N., Bailey, "
    "R., Kanan, C., Diaz, G. & Pelz, J.B. (2019). RITnet: real-time semantic "
    "segmentation of the eye for gaze tracking. In: 2019 IEEE/CVF "
    "International Conference on Computer Vision Workshop (ICCVW). IEEE, "
    "pp. 3698-3702.",

    "Ivins, J.P., Porrill, J. & Frisby, J.P. (1998). Deformable model of the "
    "human iris for measuring ocular torsion from video images. IEE "
    "Proceedings - Vision, Image and Signal Processing 145, 213-220.",

    "Lucas, B.D. & Kanade, T. (1981). An iterative image registration "
    "technique with an application to stereo vision. In: Proceedings of the "
    "7th International Joint Conference on Artificial Intelligence (IJCAI). "
    "Morgan Kaufmann, pp. 674-679.",

    "Rufer, F., Schroder, A. & Erb, C. (2005). White-to-white corneal "
    "diameter: normal values in healthy humans obtained with the Orbscan II "
    "topography system. Cornea 24, 259-261.",

    "Sadeghi, R., Ressmeyer, R., Yates, J. & Otero-Millan, J. (2024). Open "
    "Iris - an open source framework for video-based eye-tracking research and "
    "development. In: Proceedings of the 2024 Symposium on Eye Tracking "
    "Research and Applications (ETRA '24). ACM, article 21.",

    "Shi, J. & Tomasi, C. (1994). Good features to track. In: Proceedings of "
    "IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, "
    "pp. 593-600.",

    "Spearman, C. (1910). Correlation calculated from faulty data. British "
    "Journal of Psychology 3, 271-295.",
]

APPENDICES = [
    ("h2", "Appendix A. Repository contents"),
    ("p",
     "The repository at "
     "https://github.com/SxR24/Micro-analysis-of-eye-movements-using-deep-learning-neural-networks "
     "contains the following. Paths are relative to the repository root."),
    ("bullet", [
        "src/preprocess/frame_shrink.py - frame extraction with the "
        "letterbox transform and its metadata.",
        "src/ritnet/ - RITnet inference driver, the model definition and "
        "weights (third party, MIT), and the geometry measurement.",
        "src/ritnet/get_aoi.py - derivation of the region of interest from "
        "segmentation output.",
        "src/irisometry/ocular.py - torsion tracking, including reference "
        "anchoring, feature gating and the Procrustes estimator.",
        "src/irisometry/merge.py - fusion of segmentation geometry and "
        "torsion into the per-frame table.",
        "src/analysis/reliability.py - split-half reliability.",
        "src/analysis/compare_openiris.py - the method comparison.",
        "src/analysis/analyse.py and okn.py - the two control analyses.",
        "figures/make_figures.py - generates every figure in this report from "
        "the pipeline outputs, and writes figure_stats.txt listing every "
        "plotted value.",
        "run_video8.bat - reproduces the full analysis for video 8.",
    ]),
    ("h2", "Appendix B. Reproducing the reported values"),
    ("p",
     "Every statistic in Section 3 can be regenerated from the committed "
     "outputs. The reliability figures come from src/analysis/reliability.py "
     "run against the current and baseline feature archives. The comparison "
     "figures come from src/analysis/compare_openiris.py. The occlusion "
     "measurement is computed inside figures/make_figures.py from the "
     "segmentation masks and is written to figures/output/figure_stats.txt "
     "along with every other plotted value."),
    ("h2", "Appendix C. Note on citations requiring verification"),
    ("p",
     "The reference to Ivins, Porrill and Frisby (1998) should be checked "
     "against the original before submission. The work exists and is "
     "correctly attributed, but the journal, volume and page numbers given "
     "here were not confirmed against the published article."),
]

