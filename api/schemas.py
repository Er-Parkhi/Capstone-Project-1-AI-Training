
from pydantic import BaseModel, Field
 
 
class TransactionInput(BaseModel):

    step: int = Field(ge=0, description="Transaction time step")

    type: str = Field(

        description="Transaction type, e.g. TRANSFER or CASH_OUT"

    )

    amount: float = Field(ge=0)

    oldbalanceOrg: float = Field(ge=0)

    newbalanceOrig: float = Field(ge=0)

    oldbalanceDest: float = Field(ge=0)

    newbalanceDest: float = Field(ge=0)
 
 
class FraudPrediction(BaseModel):

    is_fraud: bool

    fraud_probability: float

    model_name: str

