# Friends(TV) script

## Bootstrap

- Install python dependencies
- Install nltk_data

```python
import nltk
nltk.download('punkt') # english vocab
nltk.download('averaged_perceptron_tagger') # POS tagging
```

## Data Pipeline

1. Spell Check
2. POS tagging

## Reference

- [Natural Language Pipeline for chatbots](http://pavel.surmenok.com/2016/11/05/natural-language-pipeline-for-chatbots/)
- [How to write a spelling corrector?](http://norvig.com/spell-correct.html)
- [POS Tagging with **nltk**](http://www.nltk.org/book/ch05.html)
- [POS Tags](https://cs.nyu.edu/grishman/jet/guide/PennPOS.html)
