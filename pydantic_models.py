import json
from typing import Literal, List, Union, Any, Optional

from pydantic import BaseModel, model_validator, model_serializer, Field, AliasChoices


class ASTETriplet(BaseModel):
    aspect_term: str
    opinion_term: Optional[str] = None
    sentiment: Optional[Literal['POS', 'NEG']] = None

    def model_dump_duplet(self):
        as_dict = self.model_dump()
        as_dict.pop("sentiment", None)
        return as_dict

    def model_dump_aspect(self):
        as_dict = self.model_dump()
        as_dict.pop("sentiment", None)
        as_dict.pop("opinion_term", None)
        return as_dict

    @model_validator(mode="before")
    @classmethod
    def from_literal(cls, name: Union[dict, tuple, str]) -> Any:
        if isinstance(name, dict):
            return name
        if isinstance(name, str):
            try:
                name = eval(name)
            except Exception:
                return name
        if (not isinstance(name, list) or isinstance(name, tuple)) and len(name) > 3:
            return name

        return dict(
            aspect_term=name[0],
            opinion_term=name[1] if 1 < len(name) else None,
            sentiment=name[2] if 2 < len(name) else None,
        )
    
    @model_serializer
    def _serialize(self):
        return {k: v for k, v in self if v is not None}


class ASTEAnswer(BaseModel):
    triplets: List[ASTETriplet] = Field(alias=AliasChoices("triplets", "duplets", "aspects"))

    def model_dump_duplet_json(self):
        as_dict = {}
        as_dict["duplets"] = [
            triplet.model_dump_duplet()
            for triplet in self.triplets
        ]
        return json.dumps(as_dict, ensure_ascii=False)

    def model_dump_aspect_json(self):
        as_dict = {}
        as_dict["aspects"] = [
            triplet.model_dump_aspect()
            for triplet in self.triplets
        ]
        return json.dumps(as_dict, ensure_ascii=False)
    

if __name__ == "__main__":
    answer = ASTEAnswer(triplets=[("пицца", "привет")])
    try:
        ASTEAnswer(triplets=[("pizza", "good", "garbage")])
        raise Exception
    except:
        pass
    try:
        ASTEAnswer(triplets=[("pizza", "good", "POS", "garbage")])
        raise Exception
    except:
        pass
    print(answer.model_dump_json())
    print(answer.model_dump_aspect_json())
    print(answer.model_dump_duplet_json())
    print(ASTEAnswer.model_validate({"triplets": [{"aspect_term": "a", "opinion_term": "b", "sentiment": "POS"}]}))
    print(ASTEAnswer.model_validate({"duplets": [{"aspect_term": "a", "opinion_term": "b"}]}))
    print(ASTEAnswer.model_validate({"aspects": [{"aspect_term": "a"}]}))
