# MultiGrain
MultiGrain is a neural network architecture solving both image classification and image retrieval tasks.

## Requirements
MultiGrain requires
* Python 3.5 or higher
* Pytorch 1.0 or higher

and the requirements highlighted in [requirements.txt](requirements.txt)

The requirements can be installed:
* Ether by setting up a dedicated conda environment: `conda env create -f environment.yml` followed by `source activate multigrain`
* Or with pip: `pip install -r requirements.txt`

## Usage

* Training

`scripts/train.py`...

* Input size fine-tuning of GeM exponent

`scripts/finetune_p.py` determines the optimal `p` for a given input size by fine-tuning.

* Whitening
`scripts/whiten.py` computes a PCA whitening and modifies the network accordingly.

* Evaluation
`scripts/eval.py` evaluates the network on standard benchmarks.


## Citation
Please cite the MultiGrain paper; BibTeX reference:
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

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
MultiGrain is [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) licensed, as found in the LICENSE file.
