#!/usr/bin/env python

from distutils.core import setup

setup(name='Turkish Delight NLP API',
      version='0.1',
      description="A neural Turkish NLP toolkit called TurkishDelightNLP "
                  "that performs computational linguistic analyses from morphological level "
                  "till semantic level that involves tasks such as stemming, morphological "
                  "segmentation, morphological tagging, part-of-speech tagging, dependency "
                  "parsing, and semantic parsing, as well as high-level NLP tasks such as "
                  "named entity recognition.",
      author='Huseyin Alecakir',
      author_email='huseyin.alecakir@gmail.com',
      packages=['turkishdelightnlp'],
      )

