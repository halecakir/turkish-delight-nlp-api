from fastapi import APIRouter, Depends
from starlette.requests import Request

from turkishdelightnlp.core import security
from turkishdelightnlp.common.payload import SentencePayload
from turkishdelightnlp.common.enums import JointModelSubTypes, ModelTypes
from turkishdelightnlp.common.prediction import (
    JointPredictionResult,
    POSTaggingPredictionResult,
    MorphemeSegmentationPredictionResult,
    MorphemeTaggingPredictionResult,
    DependencyPredictionResult,
)
from turkishdelightnlp.services.models import JointModel

router = APIRouter()


@router.post("/joint", response_model=JointPredictionResult, name="joint")
def joint_predict(
    request: Request,
    authenticated: bool = Depends(security.validate_request),
    block_data: SentencePayload = None,
) -> JointPredictionResult:

    model: JointModel = request.app.state.model[str(ModelTypes.JOINT)]
    prediction: JointPredictionResult = model.predict(
        block_data.sentence, submodel_type=str(
            JointModelSubTypes.JOINT))

    return prediction


@router.post(
    "/morpheme_segmentation",
    response_model=MorphemeSegmentationPredictionResult,
    name="morpheme_segmentation",
)
def morpheme_segmentation_predict(
    request: Request,
    authenticated: bool = Depends(security.validate_request),
    block_data: SentencePayload = None,
) -> MorphemeSegmentationPredictionResult:

    model: JointModel = request.app.state.model[str(ModelTypes.JOINT)]
    prediction: MorphemeSegmentationPredictionResult = model.predict(
        block_data.sentence, submodel_type=str(JointModelSubTypes.MORPH)
    )

    return prediction


@router.post(
    "/morpheme_tagging",
    response_model=MorphemeTaggingPredictionResult,
    name="morpheme_tagging",
)
def morpheme_tagging_predict(
    request: Request,
    authenticated: bool = Depends(security.validate_request),
    block_data: SentencePayload = None,
) -> MorphemeTaggingPredictionResult:

    model: JointModel = request.app.state.model[str(ModelTypes.JOINT)]
    prediction: MorphemeTaggingPredictionResult = model.predict(
        block_data.sentence, submodel_type=str(JointModelSubTypes.MTAG)
    )

    return prediction


@router.post(
    "/pos_tagging", response_model=POSTaggingPredictionResult, name="pos_tagging"
)
def pos_tagging_predict(
    request: Request,
    authenticated: bool = Depends(security.validate_request),
    block_data: SentencePayload = None,
) -> POSTaggingPredictionResult:

    model: JointModel = request.app.state.model[str(ModelTypes.JOINT)]
    prediction: POSTaggingPredictionResult = model.predict(
        block_data.sentence, submodel_type=str(JointModelSubTypes.POS)
    )

    return prediction


@router.post(
    "/dependency_parsing",
    response_model=DependencyPredictionResult,
    name="dependency_parsing",
)
def dependency_parsing_predict(
    request: Request,
    authenticated: bool = Depends(security.validate_request),
    block_data: SentencePayload = None,
) -> DependencyPredictionResult:

    model: JointModel = request.app.state.model[str(ModelTypes.JOINT)]
    prediction: DependencyPredictionResult = model.predict(
        block_data.sentence, submodel_type=str(JointModelSubTypes.DEPENDENCY)
    )

    return prediction
