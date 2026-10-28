# External validation lock

Recorded2026-09-13 Asia/Shanghai. Before external preprocessing beyond one source trial's schema/quality audit, any external classification, or any target metrics. The extension method was fixed beforehand; its Cluster results are already known and did not determine the external tuning grid. This is a local protocol record, not an external preregistration.

## Verified source and discrepancies

Lima et al., Data in Brief50(2023)109590, DOI10.1016/j.dib.2023.109590; Mendeley datasetv1 DOI10.17632/n666tk4mw9.1, CC BY4.0. Actual archive folders are30/35/40mm/s,3600IMUfiles,100episodes per12texture×3speedcells. The repository description and some article passages say30/40/45; other Methods text and the actual archive say30/35/40. Use archive labels, never relabel35as40or40as45. No recorded trajectory independently verifies instantaneous/contact speed. Related source Das et al., Frontiers in Robotics and AI(2024), DOI10.3389/frobt.2024.1281060, describes12s square exploration with3s per side; its reported experiment uses barometer data at30mm/s, so its published scores are not comparable to this IMU speed holdout.

The inspected source trial30/texture01/imu0101 has37754rows over12.025442s, nineIMUchannels and a nanosecondDatetimeIndex. Approximately92.18% of consecutive acceleration vectors repeat; observed acceleration changes occur about245.56times/s. Row count is not independent sampling rate. Physical acceleration-unit text is inconsistent with values near gravity; preserve numerical units and normalize spectral power, without an asserted unit conversion. These observations motivate a lowpass100Hz analysis, not a claim that an exact physical250Hzsensor rate has been established.

## Outcome-blind preprocessing

Read pickle payloads only after ZIPCRC/SHA and restricted passive numpy/pandas reconstruction inspection. Hash each uncompressed episode and normalized numeric/time contents; inspect exact cross-split duplicates. Retain all acquisition lineage and exclusions.

Merge exact duplicate timestamps by averaging. Interpolate full acceleration traces to1500Hz; apply sixth-order100HzButterworth zero-phase sosfiltfilt within each episode and resample_poly(1,3)to500Hz. Use four1s windows centered atT/8,3T/8,5T/8,7T/8, whereTis that episode's recorded duration. These are fractional interior-square-side windows, chosen from source protocol and one source-only trace rather than target outcomes. All four windows remain one episode and are never split across train/validation/test. Use Hann256-point Welch with128overlap, constant detrend and density scaling, sum threeaxisPSDs and average fourwindows. Frequencygrid0..250Hz,129values.

CNN inputs use the same filtered traces/windows. Perwindow STFT256/hop64, constant detrend,boundaryNone,paddedFalse;32HTKmeltriangularbands5..100Hz; peraxis/perwindow dB relative to maximum, floor−80dB, affine map to[−1,1]; concatenate fourwindows in time, bilinear resize3×128×64, float16 storage. No observation or label is added through resizing.

Exclude an entire episode for nonfinite acceleration, decreasing timestamp, duration<4.4s, or a timestamp gap>50ms intersecting one of the fouranalysiswindows. Record all reasons before fitting. Repeated values or amplitude/clipping alone are descriptive flags and are not exclusion reasons. If exclusions exceed1%or remove an entireclass/speedcell, pause before modeling to make an outcome-blind data-validity decision. If exact episode duplicates crosssource/target, the leakage issue must be repaired before proceeding and documented separately. No test accuracy may guide repair.

## Frozen model family and split

Train on all eligible episodes at nominal30/35mm/s, evaluate all eligible40mm/s episodes. Allclasses are represented in bothsets; this tests same-panel recognition across speed, not newphysicalspecimen recognition or direct transfer of a Cluster-trained classifier. Episode IDs and all derivativewindows remain grouped. Source SVM selection uses fivefolds oftrial_indexmod5, stratified implicitly by complete class/speedcells. CNN source selection uses threefolds oftrial_indexmod3 according to its separately budgeted protocol. These innerfold/search budgets differ and are explicitlyreported; the outertestset and inputepisodes are shared.

All spectral models use128logbins5..100Hz,reference speed35mm/s. The shared-response basis support is4..120Hz; lambda0.01 main,0.001and0.1sensitivities,wrong-ordercontrolaspreviouslydefined. The model coefficients are re-estimated only on this dataset'ssourcefolds. No physicalsensorresponse-identification claim. Temporal,spatial,corrected,pooledwhitening,32logmelbands5..100Hz, andDCTcoefficients1..20ofthose logmelvalues(MFCCsummary) use the samenineC/gammacandidates andsource-only scaling as the extension protocol. Narrower bands and32melbins are mandated by the source-onlyupdate-rateaudit andfiniteFFTresolution; they were fixed before external classification. CNN uses all3learningrates,≤6epochs,3fixedfinalseeds from its protocol.

## Analysis and stopping

Primary contrast correctedminusspatial macroaccuracy at40mm/s; practicalthreshold3percentagepoints plus a paired specimen-bootstrap interval excludingzero. Correctedminuslogmel is a secondarygatekept comparison only ifprimarysupportsbenefit. Spatialminustemporal andallother contrasts are descriptive. Bootstrap12specimen recalls,10000draws,seed20270913; resultsdescribe sensitivity to empiricalspecimenweighting conditional onmodels, not guaranteed coverage over futuredevices/classes/retraining. Do not treat episodecount or neural seeds as numberofphysicalspecimens.

Report every candidate,selection,checkpoint,failedrun andtargetprediction. Recordeffort andlatency. Freeze completecode/input/modelhashes beforefirsttargetinference. Negativeoutcomes donottrigger methodchanges or newtargetselection. Data acquisition uses all3600plannedIMUfiles, bounded retries/resume/concurrency and no downselection forconvenience. No additionalhuman/robotstudy, privateupload, paidcompute, contactwithauthors orsubmission.
