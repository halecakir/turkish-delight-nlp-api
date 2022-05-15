from enum import Enum


class JointModelSubTypes(Enum):
    JOINT = "joint"
    DEPENDENCY = "dependency"
    MTAG = "mtag"
    MORPH = "morph"
    POS = "pos"

    def __str__(self):
        return f"{self.value}"


class ModelTypes(Enum):
    JOINT = "JointModel"
    NER = "NER"
    STEMMER = "Stemmer"
    SEMANTIC_PARSING = "SemanticParser"

    def __str__(self):
        return f"{self.value}"
