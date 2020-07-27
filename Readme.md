# Neural Matrix Factorization (Neumf) and General Matrix Factorization (GMF) using Pytorch-lightning

This is an implementation of the paper [Neural Collaborative Filtering
](https://dl.acm.org/doi/10.1145/3038912.3052569). There are many implementation for the same paper. What is new in this one is using Pytorch lighning which allows scalability and cleaner code. Furthermore, we added the option to use BPR loss for both models. This allows users who wish to compare with these model to try different training settings.

Many of the components of this implementation (such as the model architectures, metric scripts) are taken from the following pytorch [implementation](https://github.com/yihong-chen/neural-collaborative-filtering) fore the same paper:



## Requirements and installation
Clone this repo to your local machine using https://github.com/samikhenissi/pl_neumf.git

We used pytorch 1.5.1 and pytorch-lightning==0.8.5

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements or refer to official website for [pytorch](https://pytorch.org/) and [pytorch-lightning](https://github.com/PytorchLightning/pytorch-lightning).

```bash
pandas==1.0.1
numpy==1.18.1
torch==1.5.1
pytorch-lightning==0.8.5
```
You can also use  
```bash
pip install -r requirements.txt
```

## Usage

The main training script is in train.py. You will need a training data in a pandas dataframe that has the following columns:  ['uid', 'mid', 'rating', 'timestamp']

You can try the implementation of Movielens-100K or Movielens-1m

For example, to run the training script using GMF and bpr loss on the Movielens-1m data you can use:

```bash
train.py --model GMF --data movielens1m --loss bpr 
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

To contact me with any concern you can also email me at sami.khenissi@louisville.edu
## License
[MIT](https://choosealicense.com/licenses/mit/)
