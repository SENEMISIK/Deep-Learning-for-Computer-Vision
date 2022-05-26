Optical flow is defined by the apparent motion of individual pixels on the image plane. It is a 2D velocity field, representing the apparent 2D image motion of pixels from the reference image to the target image. The task can be defined as follows: Given two images $img_1 ,img_2 \in R^{H\times W \times 3}$, the flow field $U \in R^{H\times W \times 2}$ describes the horizontal and vertical image motion between $img_1 ,img_2$.

Optical flow estimation is considered a good approximation of objects’ real physical motion projected onto the image plane. It has significant applications in the autonomous vehicle field, as it can provide 3D object motion information used for scene understanding, obstacle avoidance or path planning.

FlowNet (2015) is the first CNN model that solved the optical flow estimation problem as a supervised learning task. The FlowNet proposed in the study consists of a generic CNN network with an additional layer that represents the correlation between feature vectors in different locations in the image.
    
PWC-net (2018), outperforming FlowNet, takes a different approach on handling the two images in the Object Optical Flow problem. It first downsamples the features in several layers and warp the features of the second image to the first as a layer with learnable weights. It then passes through a cost volume layer followed by a generic CNN.
    
The state-of-art model RAFT (2020) consists of a feature extract network, a correlation layer that constructs 4D volume with inner products of all feature vector pairs for the two images, and a series of iterative update layers that take the correlation volume, the context features, and the flow prediction itself to update the flow prediction iteratively.

This repository contains 

1. RAFT-master : Codebase for the state of the art approach for optical flow estimation. Link of the paper: https://arxiv.org/pdf/2003.12039.pdf
2. models: Models that are pre-trained with RAFT
3. KITTI: Dataset (dataset, calibration files, and extension kit) Note: it doesn't have multi-view extension. More info: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow
