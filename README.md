Optical flow is defined by the apparent motion of individual pixels on the image plane. It is a 2D velocity field, representing the apparent 2D image motion of pixels from the reference image to the target image. The task can be defined as follows: Given two images $img_1 ,img_2 \in R^{H\times W \times 3}$, the flow field $U \in R^{H\times W \times 2}$ describes the horizontal and vertical image motion between $img_1 ,img_2$. \\

Optical flow estimation is considered a good approximation of objects’ real physical motion projected onto the image plane [1]. It has significant applications in the autonomous vehicle field, as it can provide 3D object motion information used for scene understanding, obstacle avoidance or path planning.

This repository contains 

1. RAFT-master : Codebase for the state of the art approach for optical flow estimation. Link of the paper: https://arxiv.org/pdf/2003.12039.pdf
2. models: Models that are pre-trained with RAFT
3. KITTI: Dataset (dataset, calibration files, and extension kit) Note: it doesn't have multi-view extension. More info: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow
