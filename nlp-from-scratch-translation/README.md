# [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

## Data Preparation

translation data between two languages is put in `./dataset/${lan1}-${lan2}.txt`

take french-english translation data as an example:

### Download

```shell
$ wget https://www.manythings.org/anki/fra-eng.zip
$ unzip fra-eng.zip && mv fra.txt eng-fra.txt
```

NOTE: fra.txt is renamed to eng-fra.txt because fra.txt stores translation pairs in the order of `eng -> fra`.

## References
* [Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
* Dataset: 
  * [Tab-delimited Bilingual Sentence Pairs from Tatoeba Project](https://www.manythings.org/anki/)
  * [Tatoeba Project](https://tatoeba.org/)
