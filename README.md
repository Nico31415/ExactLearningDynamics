# Nicolas Anguita Master Thesis

## Introduction
The primary objective of this research project is to delve into the intricate theoretical underpinnings of Deep Neural Networks (DNNs) from multiple angles. The aim is to contribute to the understanding of the dynamics of linear neural networks, investigate DNNs from an information theory perspective, and uncover novel insights into the complex behaviour of these models. The endeavour will offer a more profound comprehension of the functioning and limitations of DNNs.

## Motivation
The significance of this research lies in the widespread adoption of DNNs in various applications, ranging from computer vision to natural language processing. Despite their exceptional performance, DNNs often remain “black boxes” whose inner workings are not fully understood. A deeper understanding of DNNs is essential for improved model interpretability, robustness and trustworthiness. Additionally, the theoretical aspects of DNNs are at the core of their performance, making it very important to explore the dynamics of information-theoretic aspects for further progress.


## Research Objectives:

## Objective 1: Analytical Exploration of Weight Trajectories
Our first objected is rooted in the work of Saxe et al. [1], who provided analytical solutions to describe the trajectory of weights during the learning phase of linear neural networks. While subsequent research [2] has improved upon these solutions, some critical generalisations are yet to be incorporated. Specifically, we aim to extend the derivations to encompass scenarios involving Balanced Weights and Non Whitened Input Data. Our objective is to create a robust, generalised analytical model that transcends the limitation of previous work by relaxing the existing assumptions and validating our claims using computational methods. We also aim to look at the rate of forgetting of these networks. 
We will also apply our findings to problems like Matrix Completion, Continual Learning and Reversal Learning.

## Objective 2: Investigating the Information Dynamics of Deep Neural Networks
The second objective is grounded in the framework proposed by Tishby and Schwartz-Ziv, [3][4],  which posits a two stage learning process in deep neural networks: a “fitting” phase where layers increase information on the labels, followed by a “compression” phase where layers reduce information on the output. However, recent work by Andrew Saxe [5] has revealed counterexamples to these claims, showing that the dynamics of learning are highly dependent on factors such as the choice of activation functions and the network’s overall structure. Our objective is to perform a comprehensive analysis of these claims, delving into the conditions and mechanisms governing these transitions, including the effects of learning rate schedules and loss landscapes. Possible work could also include applying the principles of Information Bottleneck theory to the domain of continual learning, exploring how neural networks can effectively retain previously acquired knowledge while learning new tasks. This research will be done using a blend of mathematical an computational techniques.

In Andrew Saxe's paper ‘On the Information Bottleneck Theory of Deep Learning’ he and his collaborators write:

*“We emphasise that compression still may occur within a subset of the input dimensions if the task demands it. This compression, however, is interleaved rather than in a secondary phase and many not be visible by information metrics that track the overall information between a hidden layer and the input”*

One possible idea to investigate would be to find effective estimators of task-related vs non-task-related information, and utilise this to refine the analysis in this paper. We could check if they have different dynamics, try to derive some analytical and computational results etc.

A potential approach would be to consider a linear student-teacher network, partitioning the input into a set of tasks-relevant inputs X_rel and a set of task-irrelevant inputs X_irrel. This is described more in depth in section 5 of the paper (“Simultaneous Fitting and Compression”). Maybe starting with linear networks means that we could find an analytical expression for these metrics, which can then be generalised to different network architectures.



## Overall Goal: Integration for a Comprehensive Understanding
The goal in this research is to synergize these two approaches. By uniting analytical explorations of weight trajectories with investigations onto the information dynamics of deep neural networks we aspire to attain a more holistic understanding of the learning processes within these networks. 


## References
* [1] Saxe, A.M., McClelland, J.L. and Ganguli, S. (2019). A mathematical theory of semantic development in deep neural networks. Proceedings of the National Academy of Sciences, [online] 116(23), pp.11537–11546. doi:https://doi.org/10.1073/pnas.1820226116.
* [2] Braun, L., Dominé, C.C.J., Fitzgerald, J.E. and Saxe, A.M. (2022). Exact learning dynamics of deep linear networks with prior knowledge. [online] openreview.net. Available at: https://openreview.net/forum?id=lJx2vng-KiC [Accessed 24 Oct. 2023].
* [3] ieeexplore.ieee.org. (n.d.). Deep learning and the information bottleneck principle. [online] Available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7133169 [Accessed 24 Oct. 2023].
* [4] Shwartz-Ziv, R. and Tishby, N. (2017). Opening the Black Box of Deep Neural Networks via Information. [online] arXiv.org. Available at: https://arxiv.org/abs/1703.00810.
* [5] Saxe, A.M., Bansal, Y., Dapello, J., Advani, M., Kolchinsky, A., Tracey, B.D. and Cox, D.D. (2018). On the Information Bottleneck Theory of Deep Learning. [online] openreview.net. Available at: https://openreview.net/forum?id=ry_WPG-A-.


