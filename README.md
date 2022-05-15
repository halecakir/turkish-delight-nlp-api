# Turkish Delight NLP API

A neural Turkish NLP toolkit called TurkishDelightNLP that performs computational linguistic analyses from morphological level till semantic level that involves tasks such as stemming, morphological segmentation, morphological tagging, part-of-speech tagging, dependency parsing, and semantic parsing, as well as high-level NLP tasks such as named entity recognition.

## Requirements

Python 3.7

## Installation
```bash
pip install -r requirements
```

## Running

```bash
uvicorn turkishdelightnlp.main:app
```

## Run Tests

```bash
pip install tox
```

Run your tests with: 
```bash
tox
```

# References

Please cite 1 if using TurkishDelightNLP.
Cite 2 if using for one of the below tasks:
   * Joint Parsing
   * Dependency Parsing
   * Morpheme Segmentation
   * Morpheme Tagging
   * POS Tagging

Cite 3 if using for stemming and cite 4 for named entity recognition (NER). Finally, cite 5 if using for UCCA semantic parsing.

### TurkishDelightNLP: A Neural Turkish NLP Toolkit

[1] Alecakir, H., Necva, B., & Can, B. (2022). TurkishDelightNLP: A Neural Turkish NLP Toolkit. In Proceedings of NAACL 2022: Demonstrations.

```
@inproceedings{alecakir2022turkishdelightnlp,
  title={TurkishDelightNLP: A Neural Turkish NLP Toolkit},
  author={Alecakir, Huseyin and Necva, Bolucu and Can, Burcu},
  booktitle={Proceedings of NAACL 2022: Demonstrations},
  year={2022}
}
```

### Joint learning of morphology and syntax with cross-level contextual information flow

[2] Can, B., Alecakir, H., Manandhar, S. and Bozsahin, C. (in press) Joint learning of morphology and syntax with cross-level contextual information flow. Natural Language Engineering.
```
@article{can2022joint,
title={Joint learning of morphology and syntax with
cross-level contextual information flow},
author={Can Buglalilar, Burcu and
	Alecakir, Huseyin and
	Manandhar, Suresh and
	Bozsahin, Cem},
year={2022},
publisher={Cambridge University Press}}
```
### Stemming Turkish Words with LSTM Networks

[3] Burcu Can. Stemming Turkish Words with LSTM Networks. Bilişim Teknolojileri Dergisi. July 2019, Vol.12, No:3.
```
@article{burcu2019lstm,
title={Lstm aglari ile Turkce kok bulma},
author={Burcu, Can},
journal={Bilisim Teknolojileri Dergisi},
volume={12},
number={3},
pages={183--193},
year={2019}}
```

### Transfer learning for Turkish named entity recognition on noisy text

[4] Emre Kağan Akkaya and Burcu Can. Transfer Learning for Turkish Named Entity Recognition on Noisy Text. The Journal of Natural Language Engineering, Cambridge University Press. 2020
```
@article{akkaya2021transfer,
title={Transfer learning for Turkish named entity
 recognition on noisy text},
author={Akkaya, Emre Kagan and Can, Burcu},
journal={Natural Language Engineering},
volume={27},
number={1},
pages={35--64},
year={2021},
publisher={Cambridge University Press}}
```

### Self-Attentive Constituency Parsing for UCCA-based Semantic Parsing

[5] Bölücü, N., & Can, B. (2021). Self-Attentive Constituency Parsing for UCCA-based Semantic Parsing. arXiv preprint arXiv:2110.00621.
```
@article{bolucu2021self,
  title={Self-Attentive Constituency Parsing for UCCA-based Semantic Parsing},
  author={B{\"o}l{\"u}c{\"u}, Necva and Can, Burcu},
  journal={arXiv preprint arXiv:2110.00621},
  year={2021}
}
```
