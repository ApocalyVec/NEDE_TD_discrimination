# Detection of Visual Attention for target and distractor discrimination in Virtual Reality Environment
With the advent of virtual reality in neuroscience experiments, realistic human responses can be captured in natural settings for the first time. In one such experiment, EEG and pupil responses to visual attention problem are explored. As subjects move through a virtual city and are presented with different classes of images, one of which is a target, the others of which are distractors, they must count the number of target class images seen. This task is repeated multiple times over two conditions, one in which the eyes are fixed in the middle of the screen, and the other where the head may move briefly. Throughout this process, EEG data is collected via a headcap and pupil data is tracked via the VR system. 

Three deep learning architectures were tested across three different training datasets: one of which used only the EEG data, one which only used the pupil data, and one which used a combination of the two datasets. The three models include a BiLSTM with attention, an ESN, and a multi-head transformer. These models were utilized across across the 'eye' condition, the 'free' condition, and both conditions and various metrics were collected to showcase performance (ROC curves, F1, precision, recall, etc.).

## Train the network
Please contact [Zain Khan](zk2230@columbia.edu) or [Ziheng (Leo) Li](zl2990@columbia.edu) to receive access to the dataset.
Run **run.sh**, it installs all the required packages and evaluate the system with the following configurations in order:
pupil-dilation-only, EEG-only, EEG+pupil-dilation.
