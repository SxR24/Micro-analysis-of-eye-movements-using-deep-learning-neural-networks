# -*- coding: utf-8 -*-
"""
report_text_rest.py
======================================================================
Sections 3.3 onwards, written to match the voice of the earlier sections
after they were revised by the author. Sections up to and including 3.2
are held verbatim in report_text.py and are not touched by this file.
"""

RESULTS_REST = [
    ("h2", "3.3 Choice of estimator and of retirement rule"),

    ("p",
     "Two more components were varied on their own, independently of the "
     "tracking strategy. The per-feature trajectories are kept by the tracker, "
     "so the estimator could be changed without running the tracking again."),

    ("p",
     "The rotation estimator was compared in three forms on the same "
     "trajectories: the circular median of per-feature angle changes, the "
     "closed-form Procrustes rotation, and the Procrustes rotation with Tukey "
     "reweighting. The half-set correlations obtained were 0.302, 0.321 and "
     "0.306, which correspond to reliabilities of 0.464, 0.486 and 0.468. "
     "These differences are small; the reweighting did not help on this "
     "recording. Procrustes was therefore adopted on other grounds. It is the "
     "maximum-likelihood solution under the assumed noise model, its implicit "
     "weighting by squared radius matches the way in which angular precision "
     "scales, and it produces the per-frame fit residual on which Sections 3.2 "
     "and 3.7 depend. The estimator is thus not the limiting component of the "
     "pipeline."),

    ("p",
     "The rule for retiring lost features had a much bigger effect. When a "
     "correspondence was retired on its first failed forward-backward check, "
     "the median rigid-fit residual dropped from 6.32 to 2.81 px, which is "
     "what the removal of bad correspondences ought to do. However, the median "
     "number of surviving features also fell from 145 to 22, and reliability "
     "fell from 0.486 to 0.260; segments collapsed to a median length of 4.5 "
     "frames. Requiring three consecutive failures before retirement, with a "
     "feature still excluded from any frame in which it fails, keeps the "
     "benefit without the attrition. The result is a reminder that a cleaner "
     "subset is not automatically a better estimate, since precision depends "
     "on the number of observations as well as on their quality."),

    ("h2", "3.4 The result does not depend on the region of interest"),

    ("p",
     "Since the region of interest is locked for the whole run, any error in "
     "it affects every frame. Two estimates were therefore compared. The first "
     "was seeded from the first 30 valid frames and placed the circle at (449, "
     "380) with a radius of 197 px; the second was taken over the whole "
     "recording and gave (454, 407) with a radius of 191 px. The vertical "
     "displacement between the two is 27 px, which is larger than the "
     "interquartile range of the horizontal pupil position."),

    ("p",
     "Reliability was 0.768 with the first estimate and 0.772 with the second. "
     "Given 120 independent segments, the standard error on a correlation of "
     "this size is about 0.055; hence a change of 0.006 in the half-set "
     "correlation cannot be told apart from zero. Drift per segment moved in "
     "the other direction, from 0.352 to 0.377 deg."),

    ("p",
     "This null result is informative rather than disappointing. A real error "
     "of 27 px in the region of interest changed nothing, because feature "
     "detection is gated by the segmented iris mask no matter where the circle "
     "sits. The circle acts only as a coarse pre-filter and the mask does the "
     "work. Automatic derivation of the region of interest therefore does not "
     "have to be precise, and a manual tuning step required by the original "
     "implementation is removed."),

    ("h2", "3.5 Pupil tracking compared with OpenIris"),

    ("p",
     "Pupil position and size are estimated by both methods, so this arm of "
     "the comparison is available whatever happens to torsion. Both were run "
     "over the same 28,236 frames. A usable pupil was produced by OpenIris on "
     "25,005 frames (89 per cent) and by this pipeline on 25,261 frames (89 "
     "per cent), with 24,687 frames usable by both."),

    ("fig", ("fig2_pupil_comparison.png", 2,
             "Pupil tracking, this pipeline against OpenIris, on identical "
             "frames. Top: horizontal and vertical pupil position over a "
             "representative 20 s window beginning at 190 s. Bottom left: "
             "distribution of frame-to-frame change in horizontal pupil "
             "position, over all frames each method judged valid. Bottom "
             "right: autocorrelation of horizontal pupil position as a "
             "function of lag. An eye moves smoothly, so a trace whose "
             "frame-to-frame variation approaches its overall variation is not "
             "following that motion.")),

    ("p",
     "On average the two agree about where the pupil is. The median "
     "horizontal position was 454.0 px for OpenIris against 462 px here, the "
     "median vertical position 397.6 against 407 px, and the median pupil "
     "width 212.1 against 203 px. These differences are small and are in part "
     "definitional, since a segmentation centroid and an ellipse-fit centre "
     "are not the same quantity."),

    ("p",
     "Where the traces differ is in stability, and the difference is large. "
     "The standard deviation of horizontal pupil position was 13.86 px for "
     "OpenIris and 7.25 px here; the frame-to-frame standard deviation, "
     "however, was 13.61 px for OpenIris against 0.76 px here, a factor of 18. "
     "Autocorrelation at lag 1 was 0.517 against 0.995. For OpenIris the "
     "frame-to-frame variation is almost as large as the variation across the "
     "whole recording, which is the arithmetic signature of a series that is "
     "re-estimated independently on each frame rather than tracked."),

    ("p",
     "One further feature of Figure 2 is worth noting. The frame-to-frame "
     "distribution for OpenIris is bimodal, with peaks near plus and minus 9 "
     "px instead of a single peak at zero. The estimate therefore alternates "
     "between two solutions rather than scattering at random, which is "
     "consistent with an ellipse fit that has more than one stable "
     "configuration on a partially occluded pupil."),

    ("h2", "3.6 Torsion compared with OpenIris (H3)"),

    ("p",
     "Torsion was compared on the frames measured by both methods, within "
     "segments of at least 25 frames; this gave 23,950 frames in 118 "
     "segments."),

    ("fig", ("fig3_torsion_comparison.png", 3,
             "Torsion estimates from the two methods. Left and centre: the "
             "longest common segment (segment 35, 901 frames), centred within "
             "segment, from this pipeline and from OpenIris. Note the "
             "difference in vertical scale. Right: autocorrelation of each "
             "torsion series as a function of lag, computed over all 23,950 "
             "frames measured by both methods.")),

    ("p",
     "The within-segment correlation between the two was +0.03. This figure "
     "should not, however, be read as two measurements that disagree. The "
     "autocorrelation panel explains why. Torsion from this pipeline has a "
     "lag-1 autocorrelation of 0.80 and decays slowly, whereas the OpenIris "
     "series sits at 0.06 and is flat, which is what uncorrelated "
     "frame-to-frame noise looks like. The standard deviations were 0.191 deg "
     "here and 8.13 deg for OpenIris, a factor of 43."),

    ("p",
     "Enabling eyelid tracking did not change this conclusion. With the "
     "Hough-lines method the pupil detection rate rose slightly, from 87 to 89 "
     "per cent, and pupil autocorrelation improved from 0.438 to 0.517; "
     "torsion autocorrelation, however, moved only from 0.135 to 0.153. Three "
     "OpenIris configurations were run in total and none of them produced a "
     "torsion series with temporal structure (Table 1)."),

    ("table", (
        ["Configuration", "Pupil found", "Pupil x, lag-1", "Torsion, lag-1"],
        [["OpenIris, eyelid tracking off", "87 %", "0.438", "0.135"],
         ["OpenIris, Hough-lines eyelid tracking", "89 %", "0.517", "0.153"],
         ["This pipeline", "89 %", "0.995", "0.919"]],
        1,
        "Tracking stability across the three configurations run on video 8. "
        "Lag-1 autocorrelation is computed on frames each method judged valid. "
        "A series with no temporal structure has an autocorrelation near zero "
        "at this sampling rate.",
        [6.0, 3.0, 3.2, 3.3])),

    ("p",
     "For reference, a torsion series with a lag-1 autocorrelation of 0.96 was "
     "produced by OpenIris on the sample recording distributed with it. The "
     "software is therefore not broken; the failure is specific to this "
     "footage."),

    ("h2", "3.7 How much iris the cross-correlation actually sees"),

    ("p",
     "The previous section shows that correlation-based torsion fails on this "
     "recording but does not explain the reason. Since the segmentation masks "
     "label which pixels are iris, a direct measurement is possible. For "
     "twelve frames spread across the recording, the annulus sampled by "
     "OpenIris (121 to 191 px from the pupil centre) was intersected with the "
     "segmented iris and the proportion was computed."),

    ("fig", ("fig4_annulus_occlusion.png", 4,
             "Occlusion of the sampled iris annulus. Left: proportion of the "
             "annulus from 121 to 191 px that is segmented iris, for twelve "
             "frames across the recording. Centre: the same proportion "
             "resolved by angle, averaged over those frames; the polar axis "
             "follows image convention with 270 deg upward in the image. "
             "Right: mean proportion for the whole annulus and for the upper "
             "and lower sectors.")),

    ("p",
     "Averaged over the sampled frames, 59.5 per cent of the annulus is "
     "unoccluded iris; the remainder is eyelid, lash and sclera. When resolved "
     "by angle the distribution is strongly asymmetric, being 24.2 per cent in "
     "the upper sector against 80.8 per cent in the lower one. The proportion "
     "also declines through the recording, from 69.5 per cent at frame 1,000 "
     "to 51.5 per cent at frame 25,000; the eye of the subject therefore "
     "closes progressively over the 9.4 minutes."),

    ("p",
     "About two-fifths of what the cross-correlation matches against is thus "
     "tissue moving with the eyelid rather than with the eyeball. A "
     "correlation peak computed over such a signature is contaminated by a "
     "pattern that does not rotate with the eye, and in the upper sector this "
     "contamination dominates. Since this is a measurement of the recording "
     "and not an inference about the software, it accounts for the failure "
     "reported in Section 3.6. Hypothesis H3 is confirmed."),

    ("h2", "3.8 Control analyses"),

    ("p",
     "Two further analyses were run in order to check that signal is not "
     "manufactured by the pipeline where none should exist. Neither is a test "
     "of a hypothesis about this recording; both are negative controls."),

    ("h3", "Dependence on gaze direction"),

    ("p",
     "Under Listing's Law the torsional component of the rotation vector is "
     "zero for every gaze direction, and torsion measured in the image plane "
     "should therefore vary with the product of horizontal and vertical gaze "
     "angle at a slope of -0.00873 deg per deg squared. Gaze angle was derived "
     "from pupil displacement, with the measured iris used as a physical "
     "ruler."),

    ("fig", ("fig5_gaze_control.png", 5,
             "Gaze coverage and the control regression. Left: distribution of "
             "gaze positions over the recording, on a logarithmic colour "
             "scale. Right: torsion against the product of horizontal and "
             "vertical gaze angle, both centred within segment and binned into "
             "18 quantiles, with the fitted slope and the slope predicted "
             "under Listing's Law.")),

    ("p",
     "The prediction cannot be tested from this recording. Horizontal gaze "
     "varies with a standard deviation of only 1.06 deg and vertical gaze with "
     "3.35 deg, the two are correlated at r = 0.48, and 80 per cent of the "
     "frames fall into two opposing quadrants. Under these conditions the "
     "product term is close to a rescaled vertical main effect. The fitted "
     "slope was -0.0039 deg per deg squared, with a segment-level bootstrap "
     "interval of [-0.0195, +0.0024] which contains both the Listing "
     "prediction and zero."),

    ("p",
     "Read as a control, however, the result is useful. Torsion showed no "
     "dependence on gaze direction, which is what ought to be observed given "
     "that gaze is not manipulated by the design, and this indicates that the "
     "estimate is not contaminated by gaze-related translation. Resolving the "
     "prediction would require the product term to vary about four times more "
     "widely than it does here, which corresponds to a target grid spanning "
     "roughly plus or minus 8 to 10 deg in both axes."),

    ("h3", "Directional torsional drive"),

    ("p",
     "The recording was also examined for a sustained torsional response of "
     "the kind produced by a rotating visual stimulus. Slow phases were "
     "isolated by differentiating torsion within segments, marking quick "
     "phases wherever the velocity exceeded a robust threshold, and fitting a "
     "slope to each remaining run. This procedure yielded 276 slow phases with "
     "a total duration of 420.1 s."),

    ("p",
     "The duration-weighted mean slow-phase velocity was -0.037 deg per second "
     "with a bootstrap interval of [-0.053, -0.023], and the direction split "
     "was 39 per cent positive against 61 per cent negative. The interval "
     "excludes zero; the direction, however, is not consistent, and the "
     "magnitude is roughly two orders below what a torsional following "
     "response produces. No directional torsional drive is therefore present "
     "in this recording."),
]

# ======================================================================
DISCUSSION = [
    ("h2", "4.1 What the reliability figure means"),

    ("p",
     "The headline result of the methodological work is that reliability rose "
     "from 0.486 to 0.768 once tracking was anchored on the segment reference "
     "image, with the noise component of the variance falling by 60 per cent. "
     "It is worth being precise about what this number does and does not "
     "license."),

    ("p",
     "Split-half reliability asks whether the same rotation is reported by two "
     "independent halves of the iris in the same frame. A value of 0.768 means "
     "that roughly three quarters of the variance in the reported torsion is "
     "shared between them. This rules out a large class of failure, in "
     "particular the possibility that the estimate is dominated by noise "
     "specific to individual features. It does not, however, rule out a shared "
     "artefact. Iris deformation under pupil dilation would appear in both "
     "halves, and such deformation is present in this recording: fitting a "
     "similarity transform instead of a rigid rotation gives a scale term "
     "correlated with pupil diameter at r = 0.47 within segments. The "
     "reliability figure is therefore an upper bound on how well torsion "
     "specifically is measured."),

    ("p",
     "The mechanism behind the improvement should be separated from the "
     "improvement itself. Correspondence error is accumulated by chained "
     "optical flow, and the residual profile in Figure 1 shows that "
     "accumulation directly rather than by inference. The remedy is not more "
     "careful feature selection but a change in what the tracker is anchored "
     "to. Since the intensities of the reference frame are available "
     "throughout a segment, there is no reason to use the previous frame "
     "instead, and using it costs measurable precision. This point applies to "
     "any feature-based torsion tracker and does not depend on segmentation."),

    ("p",
     "The noise standard deviation of 0.106 deg is also worth placing beside "
     "the quantities it would be used to measure. Torsional drift and "
     "torsional microsaccades are reported in the literature at amplitudes of "
     "a few tenths of a degree up to a few degrees, so an instrument with a "
     "tenth of a degree of noise sits at the boundary of usefulness: adequate "
     "for the larger movements, marginal for the smallest. Stating it this way "
     "is more honest than quoting the reliability alone, and it makes clear "
     "why the remaining 1.5-fold residual growth is worth pursuing rather than "
     "accepting."),

    ("p",
     "A methodological point follows from the way all of this was established. "
     "The reliability measure was built because the conventional measures "
     "could not tell improvement apart from smoothing, and it then reported "
     "that one change helped substantially, another did nothing at all, and a "
     "third was marginal. Section 3.4 is the clearest case: an error of 27 px "
     "in the region of interest, which sounds serious, changed reliability by "
     "0.004. Without a measure that is able to return a null, that experiment "
     "would have been reported as an improvement on the strength of a jitter "
     "figure that happened to move in the right direction."),

    ("h2", "4.2 Why cross-correlation failed on this recording"),

    ("p",
     "A torsion series with a lag-1 autocorrelation of 0.96 was produced by "
     "OpenIris on its own sample footage and one of 0.06 on this recording, "
     "with identical software and a configuration adjusted for the camera "
     "geometry. The difference therefore lies in the material, and Section 3.7 "
     "quantifies it: 59.5 per cent of the sampled annulus is unoccluded iris "
     "overall, and 24.2 per cent in the upper sector."),

    ("p",
     "This matters more for correlation methods than for feature methods, and "
     "the asymmetry is structural. A feature tracker may decline to place "
     "features in an occluded sector and still estimate a rotation from those "
     "it does have, provided they are distributed widely enough in angle. A "
     "polar cross-correlation, by contrast, computes its signature over the "
     "whole annulus, so an occluded sector does not reduce the number of "
     "observations; instead it substitutes a different pattern, one that "
     "translates with the lid rather than rotating with the eye. The "
     "correlation peak is then formed partly from a signal carrying no "
     "rotational information."),

    ("p",
     "Pupil stability was improved somewhat by the Hough-lines eyelid option "
     "in OpenIris and torsion barely at all. This is consistent with lid edge "
     "detection being difficult on this footage for the same reason that the "
     "underlying problem is difficult: the lid margin is neither straight nor "
     "static, and the lashes extend well beyond it. Segmentation avoids the "
     "issue by labelling tissue rather than fitting a boundary."),

    ("p",
     "The wider claim should be scoped carefully. This is not evidence that "
     "correlation methods are inferior to feature methods in general. It is "
     "evidence that in an imaging regime where a large and varying fraction of "
     "the iris is occluded by the lid, a method sampling the whole annulus "
     "without exclusion degrades severely, and that the missing exclusion is "
     "provided by per-pixel anatomical labels. Published torsion work using "
     "correlation typically uses footage in which the eye is held open or the "
     "lid is retracted, and the present recording is not of that kind."),

    ("p",
     "The progressive decline in iris visibility, from 69.5 per cent at the "
     "start of the recording to 51.5 per cent at the end, has implications "
     "beyond this comparison. Any torsion method applied to a recording of "
     "this length works with a slowly changing amount of usable iris, so a "
     "precision figure quoted for the recording as a whole conceals a trend. "
     "This also suggests that recordings should be kept short, or that "
     "visibility should be monitored and reported alongside the measurement, "
     "which the mask makes straightforward."),

    ("h2", "4.3 Segmentation as an enabling component"),

    ("p",
     "A deep network in a hybrid pipeline is commonly treated as a replacement "
     "front end, that is, a better way of doing something the classical "
     "methods already did. The results here suggest a different reading. Of "
     "the three things supplied by segmentation to the tracker, only one, the "
     "region of interest, replaces an existing step; that step was manual and, "
     "as Section 3.4 shows, the pipeline is insensitive to getting it "
     "precisely right. The blink signal improves on an indirect inference. The "
     "per-frame iris mask has no classical equivalent at all in the inherited "
     "implementation, which lists automated lid detection as unresolved."),

    ("p",
     "The insensitivity to region-of-interest error is itself a consequence of "
     "the mask. A displacement of 27 px changed reliability by 0.004 because "
     "detection is constrained by anatomy rather than by geometry. A pipeline "
     "without mask gating would not tolerate such an error, since the circle "
     "would then be the only thing keeping features off the lid."),

    ("p",
     "The two control analyses support this reading in a way that is easy to "
     "overlook. Neither found an effect, and neither was expected to. Torsion "
     "showed no dependence on gaze direction, and no directional torsional "
     "drive is present in the recording. A pipeline generating signal from "
     "noise would be unlikely to return two clean nulls on analyses of quite "
     "different form, one a regression against a product term and the other a "
     "velocity estimate over segmented slow phases. These nulls do not "
     "demonstrate accuracy, but they do constrain the ways in which the "
     "estimate could be wrong."),

    ("h2", "4.4 Relation to published approaches"),

    ("p",
     "Video torsion measurement has followed a fairly stable pattern since the "
     "deformable iris models of the late 1990s (Ivins, Porrill and Frisby, "
     "1998). The iris annulus is located from the pupil, unwrapped into polar "
     "coordinates, reduced to a one-dimensional signature and matched against "
     "a reference by cross-correlation. A modern version of this is "
     "implemented by OpenIris, and its parameters name the steps: an angular "
     "resolution, an interpolation factor for sub-degree peak location, a "
     "Sobel filter size for the signature, and a limit on the angular search. "
     "The approach is well founded and, on the sample footage distributed with "
     "the software, it works."),

    ("p",
     "The literature on this method family generally assumes conditions which "
     "the present recording does not meet. Published torsion work typically "
     "uses footage in which the eye is held open, the lid is retracted, or the "
     "camera views the eye from an angle exposing the full annulus. The "
     "occlusion figures reported in Section 3.7, namely 59.5 per cent of the "
     "annulus and 24.2 per cent of the upper sector, describe an imaging "
     "regime outside that assumption. This is not a criticism of the published "
     "work; it is a statement about where its assumptions hold."),

    ("p",
     "The feature-tracking family, to which this pipeline belongs, is older "
     "and less commonly used for torsion. This is partly because sparse "
     "tracking on low-contrast iris stroma is difficult, and partly because "
     "the family has no principled way of deciding where features should go. "
     "It is this second problem that segmentation solves. The combination "
     "reported here is therefore not a new algorithm but a new pairing: a "
     "classical estimator whose weakness is knowing where to look, coupled to "
     "a network whose output is precisely a statement about where to look."),

    ("p",
     "Two aspects of the present implementation would apply to either family. "
     "Anchoring on the reference image rather than chaining is a property of "
     "how optical flow is invoked, and any feature-based torsion tracker would "
     "benefit from it. The split-half reliability measure is "
     "estimator-agnostic and could be computed for any method retaining "
     "per-feature trajectories. Neither depends on deep learning. The mask "
     "gating does, and it is the component for which no classical equivalent "
     "was available."),

    ("h2", "4.5 Limitations"),

    ("p",
     "Only one recording was analysed. Every result rests on video 8 and none "
     "has been replicated. Lid occlusion varies between individuals and with "
     "alertness, so the figure of 59.5 per cent is a property of this subject "
     "in this session rather than a general value. The reliability improvement "
     "is a within-recording comparison and is not exposed to this limitation "
     "in the same way, although its magnitude may not generalise. Seven "
     "further recordings are available and processing them is the obvious next "
     "step."),

    ("p",
     "No external validation was obtained. The comparison with OpenIris was "
     "intended to provide it. Because a usable torsion series was not produced "
     "by OpenIris on this footage, the outcome establishes something about the "
     "two methods under occlusion but leaves the reliability of 0.768 without "
     "independent confirmation. The estimate has not been shown to track a "
     "known rotation, and no such rotation exists in this recording."),

    ("p",
     "OpenIris was configured by the author rather than by its developers. "
     "Several parameters interact in ways that had to be established "
     "empirically, and one of them, the pupil detection threshold, behaved "
     "contrary to what pixel measurements predicted. Better performance might "
     "be achieved by someone more familiar with the software, and the "
     "comparison should be read with that possibility in mind."),

    ("p",
     "Gaze is derived from absolute pupil position, so any movement of the "
     "camera relative to the head enters the gaze estimate directly. The usual "
     "defence is a differential signal; the vertical iris centre in this "
     "implementation is however taken from the pupil centre, which makes a "
     "vertical differential measure unavailable. Torsion is not affected by "
     "this, since it is computed from feature geometry rather than from pupil "
     "position, but the gaze analysis is limited by it."),

    ("p",
     "The residual to a fitted rigid rotation still grows 1.5-fold across a "
     "segment. The accumulating component was removed by reference anchoring "
     "but not all of it. What remains is most likely the changing composition "
     "of the surviving feature set, and possibly genuine non-rigid deformation "
     "of the iris, for which the scale-pupil correlation reported in Section "
     "4.1 is evidence."),

    ("h2", "4.6 Further work"),

    ("p",
     "Three things would strengthen the present result without any new "
     "instrumentation. Processing the remaining seven recordings would "
     "establish whether the reliability figure and the occlusion measurement "
     "replicate. A recording in which gaze is directed to a two-dimensional "
     "target grid spanning approximately plus or minus 10 deg would turn the "
     "Listing analysis into a test rather than a control; the required "
     "coverage is given in Section 3.8. A stimulus driving a known torsional "
     "rotation, such as a rotating optokinetic pattern, would supply the "
     "external reference this project lacks, and the analysis code for "
     "slow-phase velocity is already written and validated on the null case."),

    ("p",
     "Two extensions to the method also suggest themselves. Restoring an "
     "independent vertical iris centre, by fitting a circle to the left and "
     "right limbus arcs where they are not clipped by the lid, would give a "
     "slip-invariant gaze vector. Implementing a polar cross-correlation "
     "estimator inside this pipeline, with the segmented iris mask used to "
     "exclude occluded sectors from the signature, would test directly whether "
     "correlation methods fail here because of the algorithm or because of the "
     "missing exclusion. That second experiment would separate the two "
     "explanations which the present comparison cannot, and it follows from "
     "the measurement in Section 3.7 rather than from speculation."),
]

# ======================================================================
CONCLUSION = [
    ("p",
     "This project set out to combine deep-learning eye segmentation with "
     "classical feature-tracking irisometry, and to establish whether the "
     "resulting estimate of ocular torsion is a measurement rather than an "
     "artefact. The pipeline was built and runs end to end on a 9.4-minute "
     "monocular recording, with published network weights used for inference "
     "only. The region of interest, the blink signal and a per-frame iris mask "
     "constraining where features may be detected are all supplied by "
     "segmentation."),

    ("p",
     "Two of the three hypotheses were tested directly and both were "
     "confirmed. Anchoring optical flow on the segment reference image, rather "
     "than chaining it from frame to frame, raised split-half reliability from "
     "0.486 to 0.768 and cut the noise standard deviation from 0.262 to 0.106 "
     "deg, with the residual to a fitted rigid rotation ceasing to grow across "
     "a segment. Compared against OpenIris on identical frames, this pipeline "
     "produced a pupil trajectory with 18 times less frame-to-frame variation "
     "and a torsion series with a lag-1 autocorrelation of 0.80 against 0.06. "
     "The reason was measured rather than assumed: only 59.5 per cent of the "
     "annulus sampled by cross-correlation is unoccluded iris, and 24.2 per "
     "cent in the upper sector."),

    ("p",
     "The third hypothesis, that mask gating improves reliability relative to "
     "a circular region of interest alone, was not tested against a matched "
     "control, for the reason given at the head of Section 3. The evidence "
     "bearing on it is indirect but consistent. Feature purity rises from 33.7 "
     "to 100 per cent once the mask is applied; the pipeline is insensitive to "
     "an error of 27 px in the region of interest, because detection is "
     "constrained by anatomy rather than geometry; and the method lacking such "
     "gating fails on the same frames for a reason that was measured rather "
     "than inferred."),

    ("p",
     "The result is best stated as a claim about conditions. Under close-up "
     "infrared imaging in which a large and varying fraction of the iris is "
     "occluded by the eyelid, torsion measurement by iris cross-correlation "
     "degrades to noise, and the anatomical exclusion which recovers a stable "
     "estimate is supplied by per-pixel segmentation. In this regime "
     "segmentation is not an alternative front end but a requirement."),

    ("p",
     "Two limitations bound that claim. All results come from a single "
     "recording and none has been replicated. External validation was not "
     "obtained, since a usable torsion series was not produced by the intended "
     "comparator on this footage; the reliability figure therefore remains an "
     "internal upper bound rather than a demonstration of accuracy. Two "
     "control analyses found no dependence of torsion on gaze direction and no "
     "directional torsional drive, which is what a recording without a "
     "rotational stimulus should show, and which indicates that signal is not "
     "generated by the pipeline where none exists."),

    ("p",
     "Beyond the specific result, a way of asking whether a torsion estimate "
     "is real was produced by this project. Split-half reliability requires no "
     "ground truth, cannot be improved by smoothing the output, and returns a "
     "null when a change does nothing, as it did for the region-of-interest "
     "experiment. It can be computed by any method that retains per-feature "
     "trajectories. Given how difficult ground truth is to obtain for torsion, "
     "and how readily the conventional internal measures reward the wrong kind "
     "of change, this seems a more useful thing to have established than any "
     "single precision figure."),

    ("p",
     "The most direct route to a stronger result is a recording containing a "
     "known torsional rotation. The analysis code for that experiment is "
     "written, validated on the null case reported here, and available in the "
     "repository."),
]
