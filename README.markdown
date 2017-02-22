# Augmented Seq2seq

![](https://img.shields.io/badge/status-wip-red.svg) ![](https://img.shields.io/badge/python-3.6-brightgreen.svg) ![](https://img.shields.io/badge/tensorflow-1.0.0-green.svg) ![](https://img.shields.io/badge/nltk-3.2.2-yellowgreen.svg)

- [x] [Vanilla Seq2seq](/vanilla.py)
- [x] [Conditioning decoder on external context](/contextual_seq2seq.py)
- [ ] Language Modeling with encoder
- [ ] Bidirectional encoder
- [ ] RNNSearch : Soft Alignment
- [ ] Memory augmentation
- [ ] Multi-turn Conversation Modeling with Hierarchical Recurrent Encoder-Decoder (HRED)

## Reference

1. Ed Grefenstette, *Beyond Sequence to Sequence with Augmented RNNs* [video](https://www.youtube.com/watch?v=4deLk3Eu05E), [slides](http://videolectures.net/site/normal_dl/tag=1051689/deeplearning2016_grefenstette_augmented_rnn_01.pdf)
2. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
3. [End-to-End Memory Networks](https://arxiv.org/abs/1503.08895)
4. [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/abs/1507.04808)
5. [Advanced Seq2seq in tensorflow](https://github.com/ematvey/tensorflow-seq2seq-tutorials)
6. [Beam Search](https://github.com/tensorflow/tensorflow/issues/654)
7. [Multi-task Learning in tensorflow](http://www.kdnuggets.com/2016/07/multi-task-learning-tensorflow-part-1.html)


## Credits

- thank you [@ematvey](https://github.com/ematvey), for helping me understand `raw_rnn`
