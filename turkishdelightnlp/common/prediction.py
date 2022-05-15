from pydantic import BaseModel


class NERPredictionResult(BaseModel):
    words: str
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
