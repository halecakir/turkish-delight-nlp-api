from pydantic import BaseModel


class SentencePayload(BaseModel):
    sentence: str
