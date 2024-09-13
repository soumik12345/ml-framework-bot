from typing import List, Optional

from pydantic import BaseModel


class Argument(BaseModel):
    arg_name: str
    description: str


class Shape(BaseModel):
    data_format: str
    shape: str


class Error(BaseModel):
    error_name: str
    condition: str


class APIReference(BaseModel):
    api_name: str
    api_signature: str
    description: str
    arguments: Optional[List[Argument]] = None
    input_shapes: Optional[List[Shape]] = None
    output_shapes: Optional[List[Shape]] = None
    return_type: Optional[str] = None
    raises: Optional[Error] = None
    example_usage: Optional[List[str]] = None


class CodeSnippets(BaseModel):
    snippets: List[str]
