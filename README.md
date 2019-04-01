# MultiGrain

MultiGrain is a neural network architecture that solves both image classification and image retrieval tasks.

The method is described in "MultiGrain: a unified image embedding for classes and instances" ([ArXiV link](https://arxiv.org/abs/1902.05509) ). 

BibTeX reference:
```bibtex
@ARTICLE{2019arXivMultiGrain,
       author = {Berman, Maxim and J{\'e}gou, Herv{\'e} and Vedaldi Andrea and
         Kokkinos, Iasonas and Douze, Matthijs},
        title = "{MultiGrain: a unified image embedding for classes and instances}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = "2019",
        month = "Feb",
          eid = {arXiv:1902.05509},
        pages = {arXiv:1902.05509},
archivePrefix = {arXiv},
       eprint = {1902.05509},
 primaryClass = {cs.CV},}
```
Please cite it if you use it. 

# Installation

The MultiGrain code requires
* Python 3.5 or higher
* Pytorch 1.0 or higher

and the requirements highlighted in [requirements.txt](requirements.txt)

The requirements can be installed:
* Ether by setting up a dedicated conda environment: `conda env create -f environment.yml` followed by `source activate multigrain`
* Or with pip: `pip install -r requirements.txt`

# Using the code 

## Extracting features with pre-trained networks

We provide pre-trained networks with ResNet-50 trunks for the following settings: 

- joint XXX 

To load a network, use the following Pytorch code: 

```
import XXY

net = xxxx()

checkpoint = torch.load('base_1B_1.0.pth')

net.load_state_dict(checkpoint['xx'])
```
The network takes images in any resolution. Preprocessing is just to remap RGB to [0, 1] and subtract XXY.

## Evaluation of the networks

To reproduce the classification results from Table 2 in the paper, run 


it should output GIST

To reproduce the image instance search results, run

it should output GIST

## Training 




# Code structure 

* Training

`scripts/train.py` trains a multigrain architecture

* Input size fine-tuning of GeM exponent

`scripts/finetune_p.py` determines the optimal p\* for a given input resolution by fine-tuning. Alternatively one may use cross-validation to determine p\*.

* Whitening
`scripts/whiten.py` computes a PCA whitening and modifies the network accordingly.

* Evaluation
`scripts/eval.py` evaluates the network on standard benchmarks.
Implementation on retrieval benchmarks is in progress, for now only the classification evaluation implemented.



## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
MultiGrain is [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) licensed, as found in the LICENSE file.

The AutoAugment implementation based on https://github.com/DeepVoltaire/AutoAugment
