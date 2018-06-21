# deepsort

Learning to sort numbers using a seq2seq model.

## Running this code

Call `pip install -r requirements.txt` to install all dependencies.

### Generating Data

All data can be generated using [`generate.py`](generate.py)

#### Sample Call

```
python generate.py \
  --name="train" \
  --size=10000 \
  --max_val=256 \
  --min_length=2 \
  --max_length=256 \
```

### Training

Models can be trained by setting appropriate parameters in [experiments/train.py](https://github.com/jacobjinkelly/deepsort/blob/master/experiments/train.py) and then setting `train.run()` to be called within [main.py](https://github.com/jacobjinkelly/deepsort/blob/master/main.py), and last calling `python main.py` (yes that is quite a mouthful, I apologize for not configuring command line arguments).

Training for 1 epoch on the dataset generated from the example call above took about 10 minutes.

### Evaluation

After training the model on `train.txt`, use [generate.py](https://github.com/jacobjinkelly/deepsort/blob/master/generate.py) to generate a test set (with `name="test"`), then run [experiments/evaluate.py](https://github.com/jacobjinkelly/deepsort/blob/master/experiments/evaluate.py) in the same way as was described for [experiments/train.py](https://github.com/jacobjinkelly/deepsort/blob/master/experiments/train.py) above to see some example evaluation of the model.

### Reproducing Results
The same as for evaluation, run [experiments/reproduce.py]() via `python main.py`, and you should see similar results (for pointer).
```
Permutation: 0.0
Nondecreasing: 0.093
```
(for vanilla attention decoder)
```
Permutation: 0.0
Nondecreasing: 0.653
```
(note: I did very minimal fine-tuning, and a very short training scheme, so it is quite likely performance could exceed this)
Change `RANDOM_SEED` in [utils.py](https://github.com/jacobjinkelly/deepsort/blob/master/utils.py/#L20) to try a different shuffle (one could also generate a new dataset and train again.)

## Issues and Improvements

Please refer to [Issues](https://github.com/jacobjinkelly/deepsort/issues) for currently known issues and areas of improvement for this code base.

## Acknowledgements

I would like to acknowledge parts of this codebase (as indicated at the top of
  file) are modified from the following sources:
* [PyTorch seq2seq tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
*  [PyTorch checkpoints](https://github.com/pytorch/examples/blob/master/imagenet/main.py)

* [Elastic Weight Consolidation](https://github.com/moskomule/ewc.pytorch)

Last, the following implementation of [Pointer Networks](
https://github.com/shiretzet/PointerNet/blob/master/PointerNet.py) was used as a guiding reference for my own implementation.
