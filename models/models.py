from sqlmodel import SQLModel, Field, Relationship


class Ingredient(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    name: str
    quantity: float
    unit: str

