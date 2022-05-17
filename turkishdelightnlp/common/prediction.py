from pydantic import BaseModel


class NERPredictionResult(BaseModel):
    ents: list
    sentence: str

class SemanticParserPredictionResult(BaseModel):
    ucca_xml: str
    sentence: str

class StemmerPredictionResult(BaseModel):
    words: list
    stems: list
    sentence: str

class JointPredictionResult(BaseModel):
    conllu: str
    morphemes: dict
    sentence: str


class DependencyPredictionResult(BaseModel):
    conllu: str
    sentence: str


class MorphemeTaggingPredictionResult(BaseModel):
    conllu: str
    sentence: str


class MorphemeSegmentationPredictionResult(BaseModel):
    morphemes: dict
    sentence: str


class POSTaggingPredictionResult(BaseModel):
    conllu: str
    sentence: str
