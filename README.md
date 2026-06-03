# IEZNAS
Imbalance-Aware Training-Free Neural Architecture Search for Clinical Images Classification

# ABSTRACT
Clinical image classification is pivotal for early disease screening and clinical decision support,
which plays crucial role in improving diagnostic efficiency. Clinical data commonly suffer from
class imbalance, wherein lesion classes contain far fewer samples than prevalent or negative
classes. Despite existing convolutional neural networks- and neural architecture search-based
methods have shown great potential in image classification by virtue of strong inductive biases
and automated architecture optimization, respectively. However, there still exists notable limita
tions, i.e., high computational cost, poor transferability and susceptibility to local optima. To this
end, we, in this paper, propose imbalance-aware training-free neural architecture search coupled
with evolutionary computation to achieve accurate clinical image classification. Specifically, we
firstly propose training-free architecture search to avoid manual architecture design to further
reduce training cost. Then, considering that it may cause weak correlation of proxy metrics
under class-imbalanced scenarios by directly transferring existing methods to medical datasets,
we design zero-cost proxy tailored for imbalanced clinical image classification. Besides, we
introduce three selection mutation with in an evolutionary framework to enhance population
diversity during architecture search and improve discovery of high-quality architectures, which
is desired to avoid entrapment in local optima. The experimental results on open datasets show
that ours effectively mitigates class imbalance and gains better classification performance and
rank consistency comparing with recent state-of-the-art methods.

<img width="4208" height="1795" alt="pipeline" src="https://github.com/user-attachments/assets/0ef623ff-5701-4662-b6ad-7b25c26eb1c8" />
## Reproduction process
### 1.Dataset preparation 
 MedMNIST ISIC Cifar10 Cifar100 Imagenet BUSI
### 2.Search space
 NAS-Bench-201
### 3.Environment configuration
 Python310 torch
