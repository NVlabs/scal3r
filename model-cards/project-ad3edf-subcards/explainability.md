# Explainability Subcard

## Intended Task/Domain:

Research

## Model Type:

Transformer

## Intended Users:

Developers and researchers building scalable online 3D reconstruction systems, such as SLAM or multi-view stereo pipelines, who need efficient multi-relative pose query capabilities. The model supports applications in robotics, augmented reality, and 3D mapping where real-time pose estimation from image streams is required.

## Output:

Other: Pointcloud

## Describe how the model works:

Scal3R reformulates online 3D reconstruction as multi-reference relative pose querying. A small set of learnable pose query tokens (~1% of parameters) is injected into a completely frozen backbone via asymmetric attention, and the predicted relative constraints are aggregated by online pose-graph optimization with keyframe selection and loop closure, suppressing long-range drift while fully preserving the backbone's pointmap quality.

## Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:

Not Applicable

## Technical Limitations and Mitigation:

This model does not perform well on low-quality input images.

## Verified to have met prescribed NVIDIA quality standards:

Not Applicable

## Performance Metrics:

Accuracy (Top-1)
F-1 Score
Throughput & Latency

## Potential Known Risks:

This model may [describe potential failure mode based on model type]

## Terms of Use/Licensing:

NVIDIA License
